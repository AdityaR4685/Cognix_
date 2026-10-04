"""Future separately authorized single-attempt CAL execution. Never run at build."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUTPUT = Path("/kaggle/working/cognix_final_conformal_cal_attempt003_v2")
URL = "https://data.carlanomaly.de/v1/carlanomaly-base-test.tar.gz"
EXPECTED_SIZE = 91538225599
EXPECTED_SHA = "267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a"


def save(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def map_segmentation_oov_to_frozen_vocabulary(array):
    """Fixed post-failure amendment: channel 0, all OOV -> existing Other=22."""
    import numpy as np
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] < 1):
        raise ValueError("FROZEN_SEGMENTATION_SCHEMA_REQUIRES_UINT8_2D_OR_3D_CHANNEL_MAP")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("FROZEN_SEGMENTATION_SCHEMA_REQUIRES_NONEMPTY_SPATIAL_MAP")
    class_map = array[..., 0] if array.ndim == 3 else array
    mapped = class_map.copy()
    mapped[mapped > 28] = 22
    return mapped


class CalProcessor:
    def __init__(self, models, agents, cal, output, device):
        self.models, self.agents, self.cal = models, agents, frozenset(cal)
        self.output, self.device = output, device
        self.current = None
        self.data = {}
        self.completed = {}
        self.total_ticks = 0

    def __call__(self, sid, relative, body):
        # Defense in depth: also gate the scientific callback itself.
        if sid not in self.cal:
            raise ValueError("SCIENTIFIC_CALLBACK_REJECTS_NON_FRESH_CAL")
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
        from calibration_adapter import labels_from_table, tick_scores
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
            raise ValueError("TOTAL_CAL_TICK_MEMORY_CAP_EXCEEDED")
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
                all_scores[scorer] = tick_scores(np.concatenate(probs), labels[1:])
        self.completed[sid] = all_scores
        ledger = {"scenario_id": sid, "partition": "FRESH_FINAL_CONFORMAL_CAL", "eligible_ticks": n - 1,
                  "tick_0_excluded": True, "scenario_scores": {k: float(v.max()) for k, v in all_scores.items()}}
        with (self.output / "cal_scenario_processing_ledger.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(ledger, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"cal_completed": len(self.completed), "elapsed_ticks": self.total_ticks}), flush=True)


def runtime_check():
    import platform
    import numpy as np
    import torch
    import PIL
    import pyarrow
    import pandas
    expected = json.loads((HERE / "runtime_lock.json").read_text())
    actual = {"Python": platform.python_version(), "NumPy": np.__version__, "PyTorch": torch.__version__,
              "Pillow": PIL.__version__, "PyArrow": pyarrow.__version__}
    for key in ("Python", "NumPy", "PyTorch"):
        if actual[key] != expected[key]:
            raise RuntimeError("FROZEN_KAGGLE_RUNTIME_MISMATCH: " + key)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "Tesla T4":
        raise RuntimeError("REQUIRES_ONE_VISIBLE_T4")
    if torch.version.cuda != "12.8" or torch.backends.cudnn.version() != 91002:
        raise RuntimeError("FROZEN_CUDA_CUDNN_MISMATCH")
    # Decoder versions are fixed during preparation from verified local schema fixtures.
    for key in ("Pillow", "PyArrow"):
        if actual[key] != expected[key]:
            raise RuntimeError("SEALED_DECODER_VERSION_MISMATCH: " + key)
    if pandas.__version__ != "2.2.3":
        raise RuntimeError("SEALED_PANDAS_VERSION_MISMATCH: pandas==2.2.3")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return actual


def require_fresh_cal_completion(completed, cal):
    if len(cal) != 100 or len(completed) != 100 or set(completed) != set(cal):
        raise ValueError("FRESH_CAL_INCOMPLETE_NO_THRESHOLDS_FINALIZED")


def execute():
    from verify_bundle import verify
    verify(require_ready=True)
    # Fixed path: no membership, checkpoint, alpha, URL, device, or output overrides.
    if OUTPUT.exists():
        raise RuntimeError("EXISTING_ATTEMPT_OR_OUTPUT_REQUIRES_SEPARATE_AUTHORIZATION")
    if HERE == OUTPUT or HERE.is_relative_to(OUTPUT) or OUTPUT.is_relative_to(HERE):
        raise RuntimeError("OUTPUT_MUST_BE_ISOLATED")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    runtime = runtime_check()
    from frozen_runtime import load
    models, agents = load("cuda:0")
    # Scientific attempt marker is created before any network access; never removed.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    save(OUTPUT / "attempt_started.json", {"single_attempt": True, "automatic_retry": False,
        "started_unix": time.time(), "runtime": runtime, "manifest_sha256": hashlib.sha256((HERE / "BUNDLE_SHA256SUMS").read_bytes()).hexdigest()})
    cal = frozenset((HERE / "final_conformal_cal.txt").read_text().splitlines())
    evaluation = frozenset((HERE / "final_evaluation.txt").read_text().splitlines())
    exclusions = frozenset(json.loads((HERE / "partition_binding.json").read_text())["historical_exclusions"])
    prior_exposed_cal = frozenset((HERE / "prior_exposed_cal.txt").read_text().splitlines())
    processor = CalProcessor(models, agents, cal, OUTPUT, "cuda:0")
    ledger = {"http_requests": 0, "eval_scenarios_decoded": 0, "cal_only_science": True, "fresh_cal_only_science": True,
              "prior_exposed_cal_scenarios_decoded": 0, "historical_excluded_scenarios_decoded": 0,
              "automatic_retry": False, "range_requests": 0, "raw_archive_retained": False}
    try:
        import urllib.request
        from streaming import scan
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                raise RuntimeError("REDIRECT_FORBIDDEN_SINGLE_REQUEST")
        opener = urllib.request.build_opener(NoRedirect())
        request = urllib.request.Request(URL, headers={"Accept-Encoding": "identity"}, method="GET")
        ledger["http_requests"] = 1
        with opener.open(request, timeout=120) as response:
            if response.status != 200 or response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("OFFICIAL_FULL_STREAM_HTTP_METADATA_MISMATCH")
            if int(response.headers.get("Content-Length", "-1")) != EXPECTED_SIZE:
                raise ValueError("OFFICIAL_COMPRESSED_LENGTH_MISMATCH")
            receipt = scan(response, cal, evaluation, exclusions, processor, ledger, EXPECTED_SIZE, prior_exposed_cal=prior_exposed_cal)
        if receipt["compressed_bytes"] != EXPECTED_SIZE or receipt["sha256"] != EXPECTED_SHA:
            raise ValueError("ARCHIVE_INTEGRITY_REFUSAL_NO_THRESHOLDS_FINALIZED")
        save(OUTPUT / "archive_verification_receipt.json", receipt)
        require_fresh_cal_completion(processor.completed, cal)
        from calibration_adapter import threshold
        thresholds, receipts, scenario_scores, diagnostics = {}, {}, {}, {}
        for scorer in models:
            rows = [{"scenario_id": sid, "score": float(processor.completed[sid][scorer].max())} for sid in sorted(cal)]
            scenario_scores[scorer] = rows
            thresholds[scorer] = threshold([r["score"] for r in rows])
            receipts[scorer] = {**thresholds[scorer], "scenario_scores_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "identical_cal_membership_sha256": hashlib.sha256((HERE / "final_conformal_cal.txt").read_bytes()).hexdigest(),
                "preregistered_score": "max_t(1-P_t(Y_t))", "method_selection_performed": False}
            q = thresholds[scorer]["Q"]
            diagnostics[scorer] = {"description": "CAL fit diagnostics only; no method ranking or generalization assessment",
                "cal_scenarios": 100, "cal_ticks": processor.total_ticks,
                "simultaneous_cal_scenarios_supported": sum(float(processor.completed[sid][scorer].max()) <= q for sid in cal)}
        save(OUTPUT / "per_scorer_scenario_scores.json", scenario_scores)
        save(OUTPUT / "conformal_thresholds.json", thresholds)
        save(OUTPUT / "threshold_calculation_receipts.json", receipts)
        save(OUTPUT / "cal_only_descriptive_diagnostics.json", diagnostics)
        save(OUTPUT / "zero_eval_decode_receipt.json", {"eval_scenarios_decoded": 0,
             "prior_exposed_cal_scenarios_decoded": 0, "historical_excluded_scenarios_decoded": 0,
             "decoder_entry_policy": "FRESH_CAL gate in scanner and processor", "limitation": "Executed-code ledger; not an OS-wide access proof"})
        save(OUTPUT / "failure_exception_audit.json", {"status": "COMPLETE", "failures": []})
        save(OUTPUT / "result_manifest.json", {"status": "COMPLETE_CAL_ONLY", "scorers": 15,
             "thresholds": 15, "cal_scenarios": 100, "eval_metrics_computed": False, "method_ranking": False})
    except BaseException as exc:
        save(OUTPUT / "failure_exception_audit.json", {"status": "INCOMPLETE_NO_RETRY", "exception": type(exc).__name__,
             "message": str(exc), "traceback": traceback.format_exc(), "cal_completed": len(processor.completed)})
        save(OUTPUT / "result_manifest.json", {"status": "INCOMPLETE", "must_not_use_partial_scores_as_final_thresholds": True})
        raise
    finally:
        ledger["cal_scenarios_completed"] = len(processor.completed)
        save(OUTPUT / "access_ledger.json", ledger)
        lines = [hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
                 for p in sorted(OUTPUT.iterdir()) if p.is_file() and not p.name.startswith("RESULT_SHA256SUMS")]
        (OUTPUT / "RESULT_SHA256SUMS").write_text("".join(lines), encoding="utf-8")
        (OUTPUT / "RESULT_SHA256SUMS.sha256").write_text(hashlib.sha256((OUTPUT / "RESULT_SHA256SUMS").read_bytes()).hexdigest() + "  RESULT_SHA256SUMS\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute-authorized-final-cal", action="store_true")
    args = parser.parse_args()
    if args.execute_authorized_final_cal and args.preflight:
        parser.error("Choose one action")
    if args.execute_authorized_final_cal:
        execute()
    else:
        from verify_bundle import verify
        print(json.dumps(verify(require_ready=True), sort_keys=True))
