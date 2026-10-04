"""Mechanically reused v4 scientific path; changed only role gate and output sink."""
import io
import json
import re
from scientific_runtime import map_segmentation_oov_to_frozen_vocabulary

class EvalProcessor:
    def __init__(self, models, agents, evaluation, output, device):
        self.models, self.agents, self.evaluation = models, agents, frozenset(evaluation)
        self.output, self.device = output, device
        self.current = None
        self.data = {}
        self.completed = {}
        self.total_ticks = 0

    def __call__(self, sid, relative, body):
        # Defense in depth: also gate the scientific callback itself.
        if sid not in self.evaluation:
            raise ValueError("SCIENTIFIC_CALLBACK_REJECTS_NON_FINAL_EVAL")
        if self.current is None:
            self.current = sid
            self.data = {"camera": {}, "seg": {}, "tables": {}}
        if sid != self.current:
            raise ValueError("SCENARIO_BUFFER_ALIGNMENT")
        if relative is None:
            self.finish(sid)
            self.current, self.data = None, {}
            return
        import numpy as np
        if relative in ("imu.feather", "anomaly-observation.feather"):
            if relative in self.data["tables"]:
                raise ValueError("DUPLICATE_REQUIRED_TABLE")
            import pyarrow.feather as feather
            columns = ["tick", "anomaly"] if relative.startswith("anomaly") else [
                "acceleration_x", "acceleration_y", "acceleration_z"]
            table = feather.read_table(io.BytesIO(body), columns=columns).to_pandas()
            # Optional sensor tick/frame alignment, when present, must also match.
            if relative == "imu.feather":
                full = feather.read_table(io.BytesIO(body))
                for name in ("tick", "frame"):
                    if name in full.column_names:
                        values = full[name].to_numpy()
                        if values.dtype.kind not in "iu" or not np.array_equal(values, np.arange(len(values))):
                            raise ValueError("IMU_EXPLICIT_INDEX_ALIGNMENT")
            self.data["tables"][relative] = table
            return
        match = re.fullmatch(r"(rgb-front)/(\d{6})\.jpg|(segmentation-front)/(\d{6})\.png", relative)
        if not match:
            raise ValueError("UNEXPECTED_REQUIRED_MODALITY_PATH")
        kind = "camera" if match.group(1) else "seg"
        tick = int(match.group(2) or match.group(4))
        if tick >= 100000 or tick in self.data[kind]:
            raise ValueError("TICK_CAP_OR_DUPLICATE_IMAGE")
        if tick == 0:
            self.data[kind][tick] = None
            return
        from PIL import Image
        from cognix.adapters.carla.real_features import camera_embedding_features, segmentation_histogram_features
        with Image.open(io.BytesIO(body)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("IMAGE_PIXEL_CAP_EXCEEDED")
            array = np.asarray(image)
            if kind == "camera":
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                    raise ValueError("FROZEN_CAMERA_SCHEMA_REQUIRES_UINT8_RGB")
                feature = camera_embedding_features(array)
            else:
                feature = segmentation_histogram_features(map_segmentation_oov_to_frozen_vocabulary(array))
        self.data[kind][tick] = feature

    def finish(self, sid):
        import numpy as np
        import torch
        from label_adapter import labels_from_table
        from cognix.adapters.carla.real_features import imu_window_features
        if sid in self.completed or set(self.data["tables"]) != {"imu.feather", "anomaly-observation.feather"}:
            raise ValueError("DUPLICATE_OR_MISSING_SCENARIO_TABLES")
        labels = labels_from_table(self.data["tables"]["anomaly-observation.feather"])
        n = len(labels)
        if n > 100000 or any(set(self.data[k]) != set(range(n)) for k in ("camera", "seg")):
            raise ValueError("MISSING_MISALIGNED_MODALITY_TICKS")
        imu = self.data["tables"]["imu.feather"].to_numpy(dtype=np.float64)
        if imu.shape != (n, 3) or not np.isfinite(imu).all():
            raise ValueError("IMU_SCHEMA_ALIGNMENT_OR_NONFINITE")
        self.total_ticks += n - 1
        if self.total_ticks > 2_000_000:
            raise ValueError("TOTAL_EVAL_TICK_MEMORY_CAP_EXCEEDED")
        nodes = np.empty((n - 1, 3, 3), dtype=np.float64)
        for t in range(1, n):
            features = {"Camera": self.data["camera"][t], "IMU": imu_window_features(imu[max(0, t - 11):t + 1]),
                        "Seg": self.data["seg"][t]}
            for i, name in enumerate(("Camera", "IMU", "Seg")):
                uq = self.agents[name].estimate_uncertainty(features[name])
                nodes[t - 1, i] = (uq.prediction, uq.epistemic, uq.aleatoric)
        if not np.isfinite(nodes).all():
            raise ValueError("NONFINITE_UPSTREAM_OUTPUT")
        all_scores = {}
        with torch.inference_mode():
            for scorer, model in self.models.items():
                probs = []
                for start in range(0, n - 1, 256):
                    batch = torch.tensor(nodes[start:start + 256], dtype=torch.float32, device=self.device)
                    e = torch.tensor(nodes[start:start + 256, :, 1], dtype=torch.float64, device=self.device)
                    probs.append(model(batch, epistemic=e).cpu().numpy())
                all_scores[scorer] = np.concatenate(probs)
        from output_store import record_scenario
        record = record_scenario(self.output, sid, labels[1:], all_scores)
        self.completed[sid] = record
        print(json.dumps({"eval_completed": len(self.completed), "elapsed_ticks": self.total_ticks}), flush=True)
