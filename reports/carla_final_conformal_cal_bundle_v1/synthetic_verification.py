"""Synthetic-only groups; contains no HTTP or TEST-file access."""
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def fake_archive(entries, fmt=tarfile.USTAR_FORMAT):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=fmt) as tf:
        for path, data in entries:
            member = tarfile.TarInfo(path)
            member.size = len(data)
            tf.addfile(member, io.BytesIO(data))
    return gzip.compress(raw.getvalue(), mtime=0)


def rejects(fn):
    try:
        fn()
    except Exception:
        return
    raise AssertionError("Expected fail-closed rejection")


def run():
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.feather as feather
    import torch
    from calibration_adapter import labels_from_table, threshold, tick_scores
    from streaming import scan
    from frozen_runtime import load, forbid
    models, agents = load()
    from cognix.adapters.carla.real_features import imu_window_features
    from cognix.adapters.carla.normality import ensemble_to_uncertainty
    from runner import CalProcessor
    from header_helpers import classify, header_fields
    results = []
    def group(name, fn):
        fn()
        results.append({"group": name, "status": "PASSED"})
    cal = {"test/normal/Town01/scenario-1"}
    evaluation = {"test/anomaly/Town02/change-weather/scenario-2"}
    excluded = {"test/anomaly/Town01/change-weather/scenario-10"}
    c, e, x = next(iter(cal)), next(iter(evaluation)), next(iter(excluded))
    def scanning(entries, fn=lambda *args: None, cals=cal, evals=evaluation, ex=excluded):
        data = fake_archive(entries)
        ledger = {}
        return scan(io.BytesIO(data), cals, evals, ex, fn, ledger, len(data)), ledger
    def opaque():
        calls = []
        entries = [(c + "/imu.feather", b"FAKE_CAL"), (e + "/imu.feather", b"INVALID_EVAL_FEATHER"),
                   (e + "/rgb-front/000001.jpg", b"INVALID_EVAL_IMAGE"),
                   (x + "/anomaly-observation.feather", b"INVALID_EXCLUDED_LABEL")]
        _, ledger = scanning(entries, lambda *args: calls.append(args))
        assert calls == [(c, "imu.feather", b"FAKE_CAL"), (c, None, None)]
        assert ledger["eval_scenarios_decoded"] == 0
    group("opaque_EVAL_and_historical_exclusion_before_decode", opaque)
    group("unknown_ID_before_body_callback", lambda: rejects(lambda: scanning([
        ("test/normal/Town01/scenario-999/imu.feather", b"X")], lambda *a: (_ for _ in ()).throw(AssertionError("decode")))))
    group("unsafe_paths_and_links", lambda: (rejects(lambda: classify("../test/normal/Town01/scenario-1/a", b"0")),
        rejects(lambda: classify("/test/normal/Town01/scenario-1/a", b"0")),
        rejects(lambda: classify("test/normal/Town01/scenario-1/../a", b"0"))))
    group("gzip_corruption_and_truncation", lambda: (rejects(lambda: scan(io.BytesIO(b"NOT_GZIP"), cal, evaluation,
          excluded, lambda *a: None, {}, 100)), rejects(lambda: scan(io.BytesIO(fake_archive([(c + "/imu.feather", b"x")])[:-9]),
          cal, set(), set(), lambda *a: None, {}, 10000))))
    group("noncontiguous_scenario_rejected", lambda: rejects(lambda: scanning([
          (c + "/imu.feather", b"x"), (e + "/imu.feather", b"x"), (c + "/rgb-front/000001.jpg", b"x")])) )
    def hashes():
        import hashlib
        data = fake_archive([(c + "/imu.feather", b"x")])
        receipt = scan(io.BytesIO(data), cal, set(), set(), lambda *a: None, {}, len(data))
        assert receipt["sha256"] == hashlib.sha256(data).hexdigest() and receipt["compressed_bytes"] == len(data)
        rejects(lambda: scan(io.BytesIO(data), cal, set(), set(), lambda *a: None, {}, len(data) - 1))
    group("incremental_full_archive_hash_and_size_bound", hashes)
    def labels():
        assert labels_from_table(pd.DataFrame({"tick": [0, 1, 2], "anomaly": [False, False, True]})).tolist() == [0, 0, 1]
        for table in ({"tick": [0, 2], "anomaly": [0, 1]}, {"tick": [0, 1], "anomaly": [0.0, 1.0]},
                      {"tick": [0, 1], "anomaly": [False, None]}, {"tick": [0, 1], "anomaly": [0, 2]}):
            rejects(lambda table=table: labels_from_table(pd.DataFrame(table)))
        assert np.allclose(tick_scores([.8, .8], [0, 1]), [.2, .8])
    group("official_tick_labels_strict_alignment_and_class_order", labels)
    def conformal():
        receipt = threshold([i / 125 for i in range(125)])
        assert receipt["rank_1_based"] == 120 and receipt["Q"] == 119 / 125
        assert threshold([])["quantile_is_infinite"] and threshold([0.1])["quantile_is_infinite"]
        assert threshold([.4] * 125)["Q"] == .4
        rejects(lambda: threshold([float("nan")]))
        scores = tick_scores([.9, .2, .8], [0, 0, 1])
        assert abs(scores.max() - .8) < 1e-14
    group("block_maximum_augmented_rank_ties_and_infinity", conformal)
    def no_training():
        for agent in agents.values():
            rejects(lambda agent=agent: agent.fit(np.zeros((2, 18))))
            rejects(lambda agent=agent: agent.fit_calibrator([0, 1], [0, 1]))
        for model in models.values():
            rejects(lambda model=model: model.train())
            assert not any(p.requires_grad for p in model.parameters())
            assert not any(child.training for child in model.modules())
    group("all_fifteen_states_strict_load_and_training_disabled", no_training)
    def uq():
        p = np.array([.1, .2, .3, .4, .5])
        value = ensemble_to_uncertainty(p)
        assert value.prediction == p.mean() and value.epistemic == max(0, value.total - value.aleatoric)
        assert abs(value.total + value.prediction * np.log(value.prediction) + (1 - value.prediction) * np.log(1 - value.prediction)) < 1e-14
        a = np.arange(45, dtype=float).reshape(15, 3)
        assert np.array_equal(imu_window_features(a[3:15]), imu_window_features(a[-12:]))
    group("frozen_entropy_decomposition_and_causal_window", uq)
    def forward():
        nodes = torch.tensor([[[.2, .01, .3], [.5, .04, .2], [.8, .1, .1]]], dtype=torch.float32)
        epistemic = nodes[:, :, 1].to(torch.float64)
        with torch.inference_mode():
            for row in json.loads((HERE / "model_registry.json").read_text())["scorers"]:
                model = models[row["scorer_id"]]
                result = model(nodes, epistemic=epistemic)
                assert torch.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
                if row["method"] != "nograph":
                    _, attention = model(nodes, epistemic=epistemic, audit_attention=True)
                    assert len(attention) == 2 and all(not torch.diagonal(a, dim1=-2, dim2=-1).any() for a in attention)
    group("frozen_scorers_synthetic_forward_attention", forward)
    def end_to_end():
        from PIL import Image
        entries = []
        n = 4
        for table_name, table in (("imu.feather", pa.table({"acceleration_x": [0., .1, .2, .3],
            "acceleration_y": [0.] * n, "acceleration_z": [9.8] * n})),
            ("anomaly-observation.feather", pa.table({"tick": list(range(n)), "anomaly": [False, False, True, False]}))):
            buffer = io.BytesIO(); feather.write_feather(table, buffer); entries.append((c + "/" + table_name, buffer.getvalue()))
        for t in range(n):
            for folder, ext, fmt, val in (("rgb-front", "jpg", "JPEG", 120), ("segmentation-front", "png", "PNG", 1)):
                buffer = io.BytesIO(); Image.fromarray(np.full((16, 16, 3), val, dtype=np.uint8)).save(buffer, format=fmt)
                entries.append((f"{c}/{folder}/{t:06d}.{ext}", buffer.getvalue()))
        entries += [(e + "/rgb-front/000001.jpg", b"DO_NOT_DECODE"), (x + "/imu.feather", b"DO_NOT_DECODE")]
        with tempfile.TemporaryDirectory(prefix="cognix_synthetic_cal_") as tmp:
            processor = CalProcessor(models, agents, cal, Path(tmp), "cpu")
            scanning(entries, processor)
            assert set(processor.completed) == cal and len(processor.completed[c]) == 15
            assert all(len(v) == 3 for v in processor.completed[c].values())
            rejects(lambda: processor(e, "imu.feather", b"X"))
    group("synthetic_archive_labels_features_fifteen_scorers_block_scores", end_to_end)
    group("incomplete_archive_scenario_set_refusal", lambda: rejects(lambda: scanning([(c + "/imu.feather", b"x")])))
    # All tests above only generated bytes in memory/tempdir. No archive URL used.
    return {"synthetic_verification_groups_passed": len(results), "synthetic_verification_groups_failed": 0,
            "groups": results, "real_TEST_bytes_used": 0}


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
