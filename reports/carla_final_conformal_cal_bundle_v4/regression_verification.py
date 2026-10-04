"""TRAIN-compatible synthetic PNG regression; no real CAL/EVAL/TEST payload."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import traceback
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent

def run():
    import numpy as np
    import pyarrow.feather as feather
    from PIL import Image
    from frozen_runtime import load
    models, agents = load()
    import cognix.adapters.carla.real_features as features
    from runner import CalProcessor
    from streaming import scan
    from synthetic_verification import fake_archive
    results = []
    def passed(name):
        results.append({"group": name, "status": "PASSED"})
    def expect_error(fn, message):
        try:
            fn()
        except ValueError as exc:
            assert str(exc) == message, (str(exc), message)
            return traceback.format_exc()
        raise AssertionError("Expected specific fail-loud schema error")
    cal, evaluation, exclusions = {"test/normal/Town01/scenario-1"}, {"test/anomaly/Town02/change-weather/scenario-2"}, {"test/anomaly/Town01/change-weather/scenario-10"}
    c, e, x = next(iter(cal)), next(iter(evaluation)), next(iter(exclusions))
    with tempfile.TemporaryDirectory(prefix="cognix_schema_regression_") as tmp:
        out = Path(tmp)
        png = (HERE / "fixtures/segmentation_2d_uint8.png").read_bytes()
        fixture_path = out / "seg.png"
        fixture_path.write_bytes(png)
        # Exact frozen cache array semantics: no convert(), channel expansion or cast.
        with Image.open(fixture_path) as image:
            train_array = np.asarray(image)
        assert train_array.dtype == np.uint8 and train_array.shape == (16, 16)
        original_feature = features.segmentation_histogram_features
        train_feature = original_feature(train_array)
        # .txt evidence is intentionally non-importable; explicit source loader for offline reproduction.
        from importlib.machinery import SourceFileLoader
        loader = SourceFileLoader("historical_v1_runner", str(HERE / "evidence/v1_runner.py.txt"))
        v1_spec = importlib.util.spec_from_loader(loader.name, loader)
        v1 = importlib.util.module_from_spec(v1_spec)
        loader.exec_module(v1)
        historical = v1.CalProcessor({}, {}, cal, out, "cpu")
        synthetic_traceback = expect_error(lambda: historical(c, "segmentation-front/000001.png", png), "FROZEN_IMAGE_SCHEMA_REQUIRES_UINT8_RGB")
        passed("historical_v1_actual_processor_rejects_2D_uint8_PNG")
        observed = []
        def spy(array):
            observed.append(array.copy())
            return original_feature(array)
        processor = CalProcessor({}, {}, cal, out, "cpu")
        with patch.object(features, "segmentation_histogram_features", spy):
            processor(c, "segmentation-front/000001.png", png)
        assert len(observed) == 1 and observed[0].dtype == train_array.dtype
        assert observed[0].shape == train_array.shape and np.array_equal(observed[0], train_array)
        assert np.array_equal(processor.data["seg"][1], train_feature)
        passed("v2_2D_uint8_PNG_exact_frozen_TRAIN_array_and_feature_semantics")
        for channels in (3, 4):
            array = np.full((16, 16, channels), 255, dtype=np.uint8)
            array[..., 0] = train_array
            b = io.BytesIO(); Image.fromarray(array).save(b, format="PNG")
            processor(c, f"segmentation-front/{channels:06d}.png", b.getvalue())
            assert np.array_equal(processor.data["seg"][channels], train_feature)
        passed("3D_segmentation_preserves_channel_zero_ignores_other_channels")
        class FakeImage:
            width = height = 16
            def __init__(self, array): self.array = array
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __array__(self, dtype=None, copy=None): return self.array
        camera_error = "FROZEN_CAMERA_SCHEMA_REQUIRES_UINT8_RGB"
        seg_error = "FROZEN_SEGMENTATION_SCHEMA_REQUIRES_UINT8_2D_OR_3D_CHANNEL_MAP"
        for array in (np.ones((16, 16), np.uint8), np.ones((16, 16, 4), np.uint8), np.ones((16, 16, 3), np.float32)):
            with patch.object(Image, "open", return_value=FakeImage(array)):
                expect_error(lambda: CalProcessor({}, {}, cal, out, "cpu")(c, "rgb-front/000001.jpg", b"synthetic"), camera_error)
        for array in (np.ones((16, 16), np.uint16), np.ones((16,), np.uint8), np.ones((1, 1, 1, 1), np.uint8), np.ones((16, 16, 0), np.uint8)):
            with patch.object(Image, "open", return_value=FakeImage(array)):
                expect_error(lambda: CalProcessor({}, {}, cal, out, "cpu")(c, "segmentation-front/000001.png", b"synthetic"), seg_error)
        passed("camera_strict_RGB_and_segmentation_separate_fail_loud_schema_errors")
        invalid = np.full((16, 16), 29, np.uint8)
        b = io.BytesIO(); Image.fromarray(invalid).save(b, format="PNG")
        amended = CalProcessor({}, {}, cal, out, "cpu")
        amended(c, "segmentation-front/000001.png", b.getvalue())
        expected = np.zeros(29, dtype=np.float64); expected[22] = 1.0
        assert np.array_equal(amended.data["seg"][1], expected)
        passed("preregistered_uniform_OOV_to_existing_Other_replaces_v2_OOV_refusal")
        # Decoders/features/labels must be unreachable for EVAL and exclusions, even with valid PNG bodies.
        calls = []
        data = fake_archive([(c + "/unneeded.bin", b"synthetic"), (e + "/segmentation-front/000001.png", png), (e + "/imu.feather", b"invalid"), (x + "/rgb-front/000001.jpg", b"invalid")])
        ledger = {}
        def forbidden(*args, **kwargs): raise AssertionError("Non-CAL decoder reached")
        with patch.object(Image, "open", forbidden), patch.object(feather, "read_table", forbidden), patch.object(features, "segmentation_histogram_features", forbidden):
            scan(io.BytesIO(data), cal, evaluation, exclusions, lambda *args: calls.append(args), ledger, len(data))
            expect_error(lambda: CalProcessor({}, {}, cal, out, "cpu")(e, "segmentation-front/000001.png", png), "SCIENTIFIC_CALLBACK_REJECTS_NON_FRESH_CAL")
        assert calls == [(c, None, None)] and ledger["eval_scenarios_decoded"] == 0
        passed("EVAL_and_exclusion_image_Feather_feature_decoder_spies_unreachable")
        import pyarrow as pa
        entries = []
        n = 4
        for name, table in (("imu.feather", pa.table({"acceleration_x": [0., .1, .2, .3], "acceleration_y": [0.] * n, "acceleration_z": [9.8] * n})),
                            ("anomaly-observation.feather", pa.table({"tick": list(range(n)), "anomaly": [False, False, True, False]}))):
            b = io.BytesIO(); feather.write_feather(table, b)
            entries.append((c + "/" + name, b.getvalue()))
        for tick in range(n):
            b = io.BytesIO(); Image.fromarray(np.full((16, 16, 3), 120, np.uint8)).save(b, format="JPEG")
            entries.extend([(f"{c}/rgb-front/{tick:06d}.jpg", b.getvalue()), (f"{c}/segmentation-front/{tick:06d}.png", png)])
        entries.extend([(e + "/segmentation-front/000001.png", png), (x + "/imu.feather", b"invalid")])
        data = fake_archive(entries)
        processor = CalProcessor(models, agents, cal, out, "cpu")
        ledger = {}
        scan(io.BytesIO(data), cal, evaluation, exclusions, processor, ledger, len(data))
        assert set(processor.completed) == cal and len(processor.completed[c]) == 15
        assert all(len(scores) == 3 and np.isfinite(scores).all() for scores in processor.completed[c].values())
        assert ledger["eval_scenarios_decoded"] == 0
        passed("2D_PNG_synthetic_archive_fifteen_frozen_scorers_end_to_end")
    return {"regression_groups_passed": len(results), "regression_groups_failed": 0, "groups": results,
            "synthetic_v1_reproduction_traceback": synthetic_traceback,
            "traceback_provenance": "Offline synthetic corroboration only; actual Attempt 001 traceback is separately available as user-supplied historical evidence",
            "real_TEST_bytes_used": 0, "TEST_network_requests": 0}

if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
