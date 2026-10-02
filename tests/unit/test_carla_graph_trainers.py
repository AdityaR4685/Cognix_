"""Frozen trainer engineering tests; no real graph training, raw input or TEST."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from cognix.adapters.carla import graph_fit_export as ge
from cognix.adapters.carla import graph_training as tr
from cognix.adapters.carla import graph_training_data as gd
from cognix.adapters.carla import graph_training_metrics as gm
from cognix.adapters.carla import graph_training_models as models

ROOT = Path(__file__).resolve().parents[2]
SEAL = ROOT / "reports/carla_gat_preregistration_v1"


@pytest.fixture(scope="module")
def splits():
    return json.loads((SEAL / "split_manifest.json").read_text())


@pytest.fixture
def fixture(splits):
    return gd.make_smoke_fixture(splits, pairs_per_scenario=8)


@pytest.fixture(scope="module")
def frozen_dataset():
    # Read/hash the real artifact, never train on it.
    return gd.load_graph_dataset(Path(os.environ["TEMP"]) / "carla_graph_fit_export_v1", SEAL)


@pytest.fixture(scope="module")
def training_runs(tmp_path_factory, splits):
    dataset = gd.make_smoke_fixture(splits, pairs_per_scenario=8)
    base = tmp_path_factory.mktemp("frozen_graph_trainer_fixtures")
    runs = {}
    for method in gd.METHODS:
        first = tr.train_model(dataset, method, 101, base / method / "first", audit_attention=True)
        second = tr.train_model(dataset, method, 101, base / method / "repeat")
        runs[method] = (first, second)
    return dataset, runs


def read_run(summary, name):
    return json.loads((Path(summary["selected_checkpoint"]).parent / name).read_text())


def test_loader_verified_scientific_hash_and_source_precision(frozen_dataset):
    assert frozen_dataset.scientific_sha256 == gd.ARTIFACT_SHA256
    assert frozen_dataset.arrays["node_features"].shape == (89970, 3, 3)
    assert frozen_dataset.arrays["node_features"].dtype == np.float64
    assert not frozen_dataset.arrays["node_features"].flags.writeable
    assert frozen_dataset.schema["node_order"] == ["Camera", "IMU", "Seg"]
    assert frozen_dataset.schema["feature_order"] == ["prob_normal", "epistemic", "aleatoric"]
    x, e, target = frozen_dataset.batch([0, 1])
    assert x.dtype == torch.float32 and e.dtype == torch.float64
    assert target.tolist() == [1, 0]
    assert frozen_dataset.arrays["node_features"].dtype == np.float64


@pytest.mark.parametrize("filename", ["protocol.json", "split_manifest.json", "manifest.json", "schema.json", "graphs.npz"])
def test_loader_hash_refusal(filename, monkeypatch):
    original = ge.sha256_file
    monkeypatch.setattr(ge, "sha256_file", lambda p: "0" * 64 if Path(p).name == filename else original(p))
    with pytest.raises(ge.GraphExportError):
        gd.load_graph_dataset(Path(os.environ["TEMP"]) / "carla_graph_fit_export_v1", SEAL)


def test_loader_scientific_hash_refusal(monkeypatch):
    monkeypatch.setattr(ge, "scientific_content_hash", lambda *a: "0" * 64)
    with pytest.raises(ge.GraphExportError, match="scientific graph hash"):
        gd.load_graph_dataset(Path(os.environ["TEMP"]) / "carla_graph_fit_export_v1", SEAL)


def test_scenario_split_isolation_and_no_cal(frozen_dataset):
    data = frozen_dataset
    train = set(data.scenarios[data.train_indices])
    val = set(data.scenarios[data.validation_indices])
    assert train == set(data.splits["GRAPH_TRAIN"]) and val == set(data.splits["GRAPH_VALIDATION"])
    assert not train & val and not (train | val) & set(data.splits["AGENT_CAL_ONLY"])
    assert len(data.train_indices) == 71976 and len(data.validation_indices) == 17994
    for pair in np.unique(data.arrays["pair_id"]):
        # Pair split isolation is already exhaustively verified by the exporter;
        # this vectorized loader-level test samples representative stable pairs.
        break
    sel = data.arrays["pair_id"] == pair
    assert len(set(data.arrays["split"][sel])) == 1


@pytest.mark.parametrize("fault", ["node_order", "feature_order", "CAL", "split", "nonfinite"])
def test_bad_dataset_contract_rejected(fixture, fault):
    arrays = {k: a.copy() for k, a in fixture.arrays.items()}
    schema = copy.deepcopy(fixture.schema)
    if fault in ("node_order", "feature_order"):
        schema[fault] = list(reversed(schema[fault]))
    elif fault == "CAL":
        schema["scenario_dictionary"][int(arrays["scenario_index"][0])]["scenario_id"] = fixture.splits["AGENT_CAL_ONLY"][0]
    elif fault == "split":
        arrays["split"][0] = 1 - arrays["split"][0]
    else:
        arrays["node_features"][0, 0, 1] = np.nan
    with pytest.raises(ge.GraphExportError):
        gd.GraphDataset(arrays, schema, fixture.splits, data_kind="synthetic_fixture")


@pytest.mark.parametrize("method,count", [("nograph", 48), ("standard_gat", 50), ("epistemic_gat", 50)])
def test_frozen_parameter_counts_and_biases(method, count):
    model = models.paired_models(101)[method]
    assert models.parameter_count(model) == count
    assert all(module.bias is None for module in model.modules() if isinstance(module, torch.nn.Linear))
    if method == "nograph":
        assert model.hidden.in_features == 3 and model.hidden.out_features == 12
        assert model.output.out_features == 1 and model.dropout.p == 0.1
        assert not hasattr(model, "adjacency")
    else:
        assert len(model.gat.layers) == 2
        assert [l.W.out_features for l in model.gat.layers] == [8, 1]
        assert all(l.leaky_relu.negative_slope == 0.2 and l.dropout.p == 0.1 for l in model.gat.layers)


@pytest.mark.parametrize("seed", gd.SEEDS)
def test_paired_initialization_is_explicit_byte_exact(seed):
    pair = models.paired_models(seed)
    for name, tensor in pair["standard_gat"].state_dict().items():
        other = pair["epistemic_gat"].state_dict()[name]
        assert tensor.numpy().tobytes() == other.numpy().tobytes()
        assert tensor.data_ptr() != other.data_ptr()


@pytest.mark.parametrize("training", [False, True])
def test_zero_E_equivalence_with_matched_rng(training):
    tr.configure_determinism(101)
    pair = models.paired_models(101)
    x = torch.full((7, 3, 3), 0.4); x[:, :, 1] = 0
    outputs = []
    for method in ("standard_gat", "epistemic_gat"):
        model = pair[method]; model.train(training)
        torch.manual_seed(8)
        outputs.append(model(x, torch.zeros((7, 3), dtype=torch.float64)))
    assert torch.equal(*outputs)


@pytest.mark.parametrize("method", ["standard_gat", "epistemic_gat"])
def test_batch_matches_unchanged_generic_forward(method):
    tr.configure_determinism(101)
    model = models.paired_models(101)[method].eval()
    raw = np.random.Generator(np.random.PCG64(1)).uniform(0.1, 0.9, (5, 3, 3))
    raw[0, :, 1] = 1e-10  # frozen float32 prior may be neutral; no amplification
    x = torch.tensor(raw, dtype=torch.float32)
    e = torch.tensor(raw[:, :, 1], dtype=torch.float64)
    actual = model(x, e).detach()
    reference = []
    for i in range(len(raw)):
        prior = model.gat.compute_epistemic_weights(dict(zip(ge.NODE_ORDER, raw[i, :, 1])), list(ge.NODE_ORDER))
        h = x[i]
        with torch.no_grad():
            for j, layer in enumerate(model.gat.layers):
                h, _ = layer(h, model.adjacency, torch.tensor(prior), final_layer=j == 1)
        reference.append(torch.sigmoid(h[:, 0]).mean())
    assert torch.allclose(actual, torch.stack(reference), atol=1e-7, rtol=1e-6)
    assert torch.allclose(actual, torch.cat([model(x[i:i+1], e[i:i+1]) for i in range(len(raw))]), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("training", [False, True])
def test_attention_incoming_normalization_no_self_edge(training):
    model = models.paired_models(101)["epistemic_gat"]
    model.train(training)
    _, attention = model(torch.full((4, 3, 3), 0.25), audit_attention=True)
    for alpha in attention:
        assert alpha.shape == (4, 3, 3)
        assert torch.isfinite(alpha).all()
        assert torch.allclose(alpha.sum(-1), torch.ones((4, 3)), atol=1e-6)
        assert not torch.diagonal(alpha, dim1=-2, dim2=-1).any()


def test_higher_E_suppresses_sender_at_both_layers_with_fixed_logits():
    model = models.paired_models(101)["epistemic_gat"].eval()
    with torch.no_grad():
        for layer in model.gat.layers:
            layer.a.weight.zero_()  # logits fixed at zero in both layers
    x = torch.full((1, 3, 3), 0.2)
    low = torch.zeros((1, 3), dtype=torch.float64)
    high = low.clone(); high[:, 0] = 2
    _, a = model(x, low, audit_attention=True)
    _, b = model(x, high, audit_attention=True)
    for first, second in zip(a, b):
        assert torch.all(second[:, 1:, 0] < first[:, 1:, 0])
        assert torch.allclose(second[:, 1:, 0], torch.full((1, 2), 0.25))


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_invalid_epistemic_rejected(value):
    model = models.paired_models(101)["epistemic_gat"]
    e = torch.zeros((1, 3), dtype=torch.float64); e[0, 0] = value
    with pytest.raises(ge.GraphExportError):
        model(torch.zeros((1, 3, 3)), e)


def test_deterministic_shuffle_and_batch_boundaries():
    data = SimpleNamespace(train_indices=np.arange(513))
    first = list(gd.epoch_batches(data, 101, 2)); second = list(gd.epoch_batches(data, 101, 2))
    assert [len(b) for b in first] == [256, 256, 1]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    expected = np.random.Generator(np.random.PCG64(103)).permutation(513)
    assert np.array_equal(np.concatenate(first), expected)
    assert not np.array_equal(expected, np.concatenate(list(gd.epoch_batches(data, 101, 3))))


def test_standard_epistemic_same_batches_and_train_isolation(fixture):
    models.paired_models(101)
    standard = list(gd.epoch_batches(fixture, 101, 3))
    epi = list(gd.epoch_batches(fixture, 101, 3))
    assert all(np.array_equal(a, b) for a, b in zip(standard, epi))
    assert all(np.all(fixture.arrays["split"][b] == 0) for b in standard)
    assert not set(np.concatenate(standard)) & set(fixture.validation_indices)


def test_BCE_normal_target_direction():
    good = F.binary_cross_entropy(torch.tensor([0.99, 0.01]), torch.tensor([1.0, 0.0])).item()
    bad = F.binary_cross_entropy(torch.tensor([0.01, 0.99]), torch.tensor([1.0, 0.0])).item()
    assert good < bad
    assert gm.bce([0.99, 0.01], [1, 0]) == pytest.approx(good, abs=1e-7)


def test_scenario_macro_not_pooled_controls_checkpoint():
    scenarios = np.array(["a"] * 100 + ["b", "c"])
    targets = np.ones(102)
    first = np.exp(-np.array([0.1] * 100 + [1.0, 1.0]))
    second = np.exp(-np.array([0.2] * 100 + [0.5, 0.5]))
    m1, _ = gm.scenario_macro_bce(first, targets, scenarios)
    m2, _ = gm.scenario_macro_bce(second, targets, scenarios)
    assert m2 < m1 and gm.bce(second, targets) > gm.bce(first, targets)
    stop = tr.EarlyStopping(); stop.observe(m1, 1)
    save, _ = stop.observe(m2, 2)
    assert save and stop.best_epoch == 2


def test_early_stopping_dual_trackers_patience_and_min_delta():
    stop = tr.EarlyStopping()
    assert stop.observe(0.5, 1) == (True, False)
    # Smaller-than-min_delta improvements save actual minima, without resetting patience.
    for i in range(1, 11):
        save, halt = stop.observe(0.5 - i * 0.0000001, i + 1)
        assert save and halt == (i == 10)
    assert stop.best_epoch == 11 and stop.tracked_best == 0.5 and stop.bad_epochs == 10


def test_earliest_exact_tie_and_strict_min_delta():
    stop = tr.EarlyStopping(); stop.observe(0.5, 1)
    assert stop.observe(0.5, 2) == (False, False)
    assert stop.best_epoch == 1
    stop.observe(0.5 - 1e-5, 3)
    assert stop.tracked_best == 0.5 and stop.best_epoch == 3
    stop.observe(0.49, 4)
    assert stop.bad_epochs == 0 and stop.tracked_best == 0.49


def test_AUROC_orientation_AP_and_ties():
    metric = gm.binary_metrics([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0], 0.5)
    assert metric["AUROC"] == metric["AUPRC"] == metric["F1"] == 1
    tied = gm.binary_metrics([0.5, 0.5], [1, 0], 0.5)
    assert tied["AUROC"] == tied["AUPRC"] == 0.5
    assert gm.binary_metrics([0.1, 0.9], [1, 0], 0.5)["AUROC"] == 0


def test_threshold_largest_exact_tie_and_boundary():
    r = gm.select_validation_threshold([0.5] * 6, [1, 0] * 3, ["a", "a", "b", "b", "c", "c"], [1] * 6)
    assert r["threshold"] == 0.5 and r["validation_scenario_macro_F1"] == pytest.approx(2 / 3)
    assert gm.select_validation_threshold([1, 0], [1, 0], ["a", "a"], [1, 1])["threshold"] == 1
    assert gm.select_validation_threshold([0.5, 0.5], [1, 1], ["a", "a"], [1, 1])["threshold"] == 1


def test_training_Cal_threshold_rejected():
    with pytest.raises(ge.GraphExportError):
        gm.select_validation_threshold([0.5, 0.5], [1, 0], ["a", "a"], [0, 0])
    with pytest.raises(ge.GraphExportError):
        gm.select_validation_threshold([0.5, 0.5], [1, 0], ["a", "a"], [2, 2])


def test_ECE_exact_boundaries_zero_one_and_empty_bins():
    q = np.arange(16, dtype=float) / 15
    _, bins = gm.ece15(q, np.zeros(16))
    assert [b["count"] for b in bins] == [1] * 14 + [2]
    assert bins[0]["mean_probability"] == 0
    assert bins[14]["mean_probability"] == pytest.approx((14 / 15 + 1) / 2)
    zero, empty = gm.ece15([0], [0])
    assert zero == 0 and all(b["weighted_gap"] == 0 and b["mean_probability"] is None for b in empty[1:])
    one, last = gm.ece15([1], [0])
    assert one == 1 and last[-1]["count"] == 1
    assert gm.ece15([0.2, 0.7], [0, 1])[0] == pytest.approx(0.25)


def test_undefined_metrics_null_with_reason_and_no_dropped_scenarios():
    r = gm.binary_metrics([0.7, 0.9], [1, 1], 0.5)
    assert r["AUROC"] is None and r["AUPRC"] is None and r["balanced_accuracy"] is None
    assert set(r["undefined_reasons"]) == {"AUROC", "AUPRC", "balanced_accuracy"}
    grouped = gm.grouped_metrics([0.7, 0.9, 0.9, 0.1], [1, 1, 1, 0], ["a", "a", "b", "b"], 0.5)
    assert grouped["scenario_macro"]["AUROC"] is None
    assert "a" in grouped["scenario_macro"]["undefined_reasons"]["AUROC"]


@pytest.mark.parametrize("method", gd.METHODS)
def test_smoke_loss_gradients_updates_validation_and_exact_repeat(training_runs, method):
    dataset, runs = training_runs
    a, b = runs[method]
    ha, hb = read_run(a, "history.json"), read_run(b, "history.json")
    assert a["finite_gradients"] and a["parameters_updated"] and a["optimizer_updates"] > 0
    assert ha[-1]["train_BCE"] < ha[0]["train_BCE"]
    assert ha == hb and a["selected_epoch"] == b["selected_epoch"]
    assert a["checkpoint_content_sha256"] == b["checkpoint_content_sha256"]
    assert a["prediction_content_sha256"] == b["prediction_content_sha256"]
    assert read_run(a, "metrics.json") == read_run(b, "metrics.json")
    for row in ha:
        assert set(row["validation_per_scenario_BCE"]) == set(dataset.splits["GRAPH_VALIDATION"])
        assert np.all(dataset.arrays["split"][row["train_row_order"]] == 0)


def test_zero_E_training_trajectories_equal(training_runs):
    _, runs = training_runs
    standard, epi = runs["standard_gat"][0], runs["epistemic_gat"][0]
    assert standard["initial_model_content_sha256"] == epi["initial_model_content_sha256"]
    assert read_run(standard, "history.json") == read_run(epi, "history.json")
    assert standard["prediction_content_sha256"] != epi["prediction_content_sha256"]  # method metadata differs
    with np.load(Path(standard["selected_checkpoint"]).parent / "predictions.npz") as a, np.load(Path(epi["selected_checkpoint"]).parent / "predictions.npz") as b:
        assert np.array_equal(a["q_normal"], b["q_normal"])


@pytest.mark.parametrize("method", gd.METHODS)
def test_checkpoint_restore_and_row_keyed_predictions(training_runs, method):
    dataset, runs = training_runs
    r = runs[method][0]
    model, optimizer, payload = tr.load_checkpoint(r["selected_checkpoint"], dataset)
    q = tr.evaluate(model, dataset)
    with np.load(Path(r["selected_checkpoint"]).parent / "predictions.npz") as z:
        assert np.array_equal(z["q_normal"], q)
        assert np.array_equal(z["p_corrupt"], 1 - q)
        for key in ("row_key", "pair_id", "tick", "target_normal", "split"):
            assert np.array_equal(z[key], dataset.arrays[key])
        assert np.array_equal(z["scenario"], dataset.scenarios)
    assert payload["metadata"]["data_kind"] == "synthetic_fixture"
    assert payload["metadata"]["graph_artifact_scientific_sha256"] == gd.ARTIFACT_SHA256
    assert optimizer.param_groups[0]["lr"] == 0.001 and optimizer.param_groups[0]["weight_decay"] == 0.0001


def test_checkpoint_continuation_exact_trajectory(training_runs, tmp_path):
    dataset, runs = training_runs
    reference = runs["standard_gat"][0]
    path = Path(reference["selected_checkpoint"]).parent / "checkpoint_epoch_050.pt"
    assert path.exists()
    continued = tr.train_model(dataset, "standard_gat", 101, tmp_path / "continued", resume_checkpoint=path)
    assert read_run(continued, "history.json") == read_run(reference, "history.json")
    assert continued["checkpoint_content_sha256"] == reference["checkpoint_content_sha256"]


def test_checkpoint_cannot_overwrite_and_dependency_tamper(training_runs, monkeypatch):
    dataset, runs = training_runs
    r = runs["nograph"][0]
    with pytest.raises(FileExistsError):
        tr.train_model(dataset, "nograph", 101, Path(r["selected_checkpoint"]).parent)
    monkeypatch.setattr(tr, "trainer_identity", lambda: {"version": "tampered"})
    with pytest.raises(ge.GraphExportError, match="dependency"):
        tr.load_checkpoint(r["selected_checkpoint"], dataset)


def test_per_recipe_metrics_pair_clean_with_pseudo(fixture):
    q = np.where(fixture.arrays["target_normal"] == 1, 0.9, 0.1)
    report = gm.dataset_metric_report(fixture, q)
    for split, data in report["per_split"].items():
        for recipe, v in data["per_recipe"].items():
            assert v["pooled"]["AUROC"] == 1
            assert v["pooled"]["confusion"]["tp"] == v["pooled"]["confusion"]["tn"]


def test_no_TEST_raw_loader_or_CAL_code_path():
    source = Path(gd.__file__).read_text()
    assert "CarlAnomalyLoader" not in source and "load_frame(" not in source and "load_timestep_labels" not in source
    assert "np.random" not in source.split("def load_graph_dataset", 1)[1].split("def epoch_batches", 1)[0]


def test_full_training_protected_before_model_creation(frozen_dataset, monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "paired_models", lambda *a: pytest.fail("full run reached model construction"))
    with pytest.raises(ge.GraphExportError, match="protected"):
        tr.train_model(frozen_dataset, "standard_gat", 101, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_oversized_full_data_cannot_be_relabeled_smoke(frozen_dataset, monkeypatch, tmp_path):
    data = copy.copy(frozen_dataset)
    data.data_kind = "synthetic_fixture"
    monkeypatch.setattr(tr, "paired_models", lambda *a: pytest.fail("oversized fixture reached models"))
    with pytest.raises(ge.GraphExportError, match="oversized"):
        tr.train_model(data, "standard_gat", 101, tmp_path / "oversized")
    assert not (tmp_path / "oversized").exists()


def test_generic_graph_source_hash_refusal(monkeypatch):
    original = ge.sha256_file
    monkeypatch.setattr(ge, "sha256_file", lambda p: "0" * 64 if Path(p).name == "epistemic_gat.py" else original(p))
    with pytest.raises(ge.GraphExportError, match="generic graph source"):
        gd.load_graph_dataset(Path(os.environ["TEMP"]) / "carla_graph_fit_export_v1", SEAL)


def test_future_environment_requires_CUDA_fixture_evidence():
    lock = tr.proposed_environment_lock()
    lock["validated_execution_environment"] = lock["local_observed"]
    with pytest.raises(ge.GraphExportError, match="CUDA smoke"):
        tr.verify_environment_lock(lock, "cpu")


def test_full_command_requires_explicit_execution(tmp_path):
    output = tmp_path / "forbidden_cli"
    result = subprocess.run([sys.executable, str(ROOT / "reports/carla_graph_trainers_v1/train.py"),
        "train", "--method", "standard_gat", "--seed", "101", "--output", str(output)], capture_output=True, text=True)
    assert result.returncode != 0 and "explicitly authorized future full execution" in result.stderr
    assert not output.exists()


def test_environment_lock_real_versions_and_CUDA_not_invented():
    lock = tr.proposed_environment_lock()
    local = lock["local_observed"]
    assert local["Python"] == __import__("platform").python_version()
    assert local["PyTorch"] == torch.__version__ and local["NumPy"] == np.__version__
    assert lock["Kaggle_observed_versions"] is None and not lock["full_training_authorized_this_milestone"]
    assert local["deterministic_algorithms"] and not local["matmul_TF32"] and not local["cuDNN_TF32"]
    assert local["cuDNN_deterministic"] and not local["cuDNN_benchmark"]
    with pytest.raises(ge.GraphExportError, match="observed, validated"):
        tr.verify_environment_lock(lock, "cpu")


def test_frozen_paired_statistics_missing_seed_and_zero_SD():
    assert not gm.paired_difference_summary({101: 0.1})["complete"]
    summary = gm.paired_difference_summary(dict.fromkeys(gd.SEEDS, 0.0))
    assert summary["exact_two_sided_sign_flip_p"] == 1 and summary["paired_dz"] is None
    positive = gm.paired_difference_summary(dict.fromkeys(gd.SEEDS, 0.1))
    assert positive["exact_two_sided_sign_flip_p"] == 0.0625
    bootstrap = gm.scenario_bootstrap(np.ones((5, 3)) * 0.2)
    assert bootstrap["percentile_interval"] == pytest.approx([0.2, 0.2])
