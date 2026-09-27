import pytest
import numpy as np
from cognix.evaluation.scenarios import DataGenerator
from experiments.universal_evaluation.run_universal_benchmark import run_evaluation

def test_noise_sweep_data_generation_scaling():
    """Verify that NOISE_SWEEP severity levels actually corrupt feature 1 and scale with severity."""
    seed = 42
    gen = DataGenerator(n_samples=50, seed=seed)
    X_base, y_base = gen.generate_base(num_features=4)

    noise_levels = [0.0, 1.0, 3.0, 5.0]
    mean_diffs = []
    max_diffs = []

    for s in noise_levels:
        # Mimic the condition routing logic in run_evaluation
        cond = "HIGH_NOISE" if s > 0.0 else "NORMAL"
        X_cond, y_cond, ood_labels = gen.apply_condition(
            X_base, y_base, condition=cond, noise_level=s
        )
        
        diff_feat1 = np.abs(X_cond[:, 1] - X_base[:, 1])
        diff_other = np.abs(X_cond[:, [0, 2, 3]] - X_base[:, [0, 2, 3]])
        
        # Unperturbed features must remain exactly identical
        assert np.max(diff_other) == 0.0, "Features other than column 1 were unexpectedly modified"

        mean_diff = np.mean(diff_feat1)
        max_diff = np.max(diff_feat1)
        mean_diffs.append(mean_diff)
        max_diffs.append(max_diff)

    # Severity 0.0 must correspond to clean data
    assert mean_diffs[0] == 0.0, "Severity 0.0 should have 0 injected noise"
    assert max_diffs[0] == 0.0, "Severity 0.0 should have 0 max injected noise"

    # Nonzero severities must actually change the feature
    assert mean_diffs[1] > 0.0, "Severity 1.0 failed to inject noise"
    assert mean_diffs[2] > 0.0, "Severity 3.0 failed to inject noise"
    assert mean_diffs[3] > 0.0, "Severity 5.0 failed to inject noise"

    # Larger severity produces strictly larger injected corruption under the same seed
    assert mean_diffs[1] < mean_diffs[2] < mean_diffs[3], (
        f"Mean diffs not strictly scaling with noise level: {mean_diffs}"
    )
    assert max_diffs[1] < max_diffs[2] < max_diffs[3], (
        f"Max diffs not strictly scaling with noise level: {max_diffs}"
    )


def test_noise_sweep_run_evaluation_pipeline():
    """Verify that run_evaluation with scenario_name='NOISE_SWEEP' records scaling corruption."""
    seed = 42
    results = {}
    for s in [0.0, 1.0, 3.0, 5.0]:
        res = run_evaluation(
            seed=seed,
            graph_type="NoGraph",
            scenario_name="NOISE_SWEEP",
            noise_level=s,
            num_agents=4,
            n_train=20,
            n_cal=20,
            n_test=30,
        )
        results[s] = res

    # Check recorded corruption
    assert results[0.0]["injected_noise_mean"] == 0.0, "noise_level=0.0 should have 0 injected noise"
    assert results[1.0]["injected_noise_mean"] > 0.0, "noise_level=1.0 should have nonzero noise"
    assert results[3.0]["injected_noise_mean"] > results[1.0]["injected_noise_mean"], (
        "noise_level=3.0 should have higher corruption than 1.0"
    )
    assert results[5.0]["injected_noise_mean"] > results[3.0]["injected_noise_mean"], (
        "noise_level=5.0 should have higher corruption than 3.0"
    )
