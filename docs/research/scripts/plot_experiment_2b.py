"""Plot already-authenticated Experiment 2B differences; never rerun scoring.

Only frozen JSON summaries and seal inventories are read. No model, label, or
prediction loading is part of reporting. Outputs stay in the report directory.
Run from any directory with: python <repository>/docs/research/scripts/plot_experiment_2b.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "reports" / "carla_train_full_v1"
OUTPUT = ROOT / "docs" / "research" / "figures" / "experiment_2b"
CONCLUSION_SEAL = "cff86433be91a3fdabd3b43568b7fb2c24c0760fa78fddeb298b9ae9cbb0cad8"
CONCLUSION_SHA256 = "6abef23828dd6a05fa05ab7e56e8be74072d5674417aafe9e321ee959b0e2f33"
PHASE_B_SEAL = "ec5ea056cb53050c284816c470845f233d4fe8c566917195658639d20bdee15e"
COMPARISONS = (
    ("StandardGAT_minus_NoGraph", "StandardGAT − NoGraph", "#17618c", "o", "-"),
    ("EpistemicGAT_minus_StandardGAT", "EpistemicGAT − StandardGAT", "#ad5213", "D", "--"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed_file(bundle: Path, expected_seal: str, name: str) -> Path:
    """Authenticate the seal inventory and requested JSON without reading labels."""
    inventory = bundle / "SHA256SUMS"
    if sha256(inventory) != expected_seal:
        raise ValueError(f"Seal mismatch: {inventory}")
    if (bundle / "SHA256SUMS.sha256").read_text(encoding="utf-8").split()[0] != expected_seal:
        raise ValueError(f"Detached seal mismatch: {bundle}")
    entries = dict(line.split("  ", 1)[::-1] for line in inventory.read_text(encoding="utf-8").splitlines())
    target = bundle / name
    if name not in entries or sha256(target) != entries[name]:
        raise ValueError(f"Payload hash mismatch: {target}")
    return target


def authenticated_data() -> dict:
    conclusion_path = sealed_file(
        EXPERIMENT / "graph_test_final_conclusion_bundle_v1", CONCLUSION_SEAL, "final_conclusion.json"
    )
    if sha256(conclusion_path) != CONCLUSION_SHA256:
        raise ValueError("Final conclusion identity differs from the authenticated reporting identity")
    metrics_path = sealed_file(EXPERIMENT / "graph_test_phase_B_results_v1", PHASE_B_SEAL, "metrics.json")
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if conclusion["phase_B_result_seal"] != PHASE_B_SEAL:
        raise ValueError("Conclusion is bound to a different Phase-B result")
    if conclusion["test_scope"] != {"runs": 15, "scenarios": 627, "scored_rows": 187473, "seeds": [101, 202, 303, 404, 505]}:
        raise ValueError("Unexpected frozen TEST scope")
    paired = {}
    for key, *_ in COMPARISONS:
        paired[key] = {}
        for metric in ("BCE", "F1"):
            summary = conclusion["paired_effects"][key][metric]
            if summary != metrics["paired_comparisons"][metric][key]:
                raise ValueError(f"Stored paired summaries disagree: {key}/{metric}")
            paired[key][metric] = summary
        if "mean" in conclusion["paired_effects"][key]["AUROC"]:
            raise ValueError("Paired macro-AUROC must remain unavailable")
    if any(row["AUROC"] is not None for row in conclusion["seed_mean_metrics"].values()):
        raise ValueError("Prespecified macro-AUROC must remain undefined")
    return {
        "schema": "COGNIX-Experiment-2B-report-figure-data-v1",
        "source": {
            "conclusion": conclusion_path.relative_to(ROOT).as_posix(),
            "conclusion_sha256": CONCLUSION_SHA256,
            "conclusion_bundle_seal": CONCLUSION_SEAL,
            "phase_B_metrics": metrics_path.relative_to(ROOT).as_posix(),
            "phase_B_metrics_sha256": sha256(metrics_path),
            "phase_B_result_seal": PHASE_B_SEAL,
        },
        "seeds": conclusion["test_scope"]["seeds"],
        "seed_mean_metrics": conclusion["seed_mean_metrics"],
        "paired_effects": paired,
        "metric_directions": {"BCE": "lower is better; negative delta favorable", "F1": "higher is better; positive delta favorable"},
        "plot": {"size_inches": [3.5, 2.8], "png_dpi": 600, "y_limits": [-0.045, 0.009], "zero_reference_line": True},
        "metric_recomputation": False,
        "raw_TEST_labels_opened": False,
    }


def main() -> None:
    data = authenticated_data()  # Fail before creating output if source authentication fails.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cognix_2b_report_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "svg.fonttype": "none",
        "svg.hashsalt": "COGNIX-closed-Experiment-2B",
        "savefig.facecolor": "white",
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "figure_data.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seeds = data["seeds"]
    for metric in ("BCE", "F1"):
        fig, ax = plt.subplots(figsize=data["plot"]["size_inches"], layout="constrained")
        for key, label, color, marker, linestyle in COMPARISONS:
            values = [data["paired_effects"][key][metric]["per_seed"][str(seed)] for seed in seeds]
            ax.plot(range(len(seeds)), values, label=label, color=color, marker=marker,
                    linestyle=linestyle, linewidth=1.15, markersize=3.8, markeredgewidth=0.6)
        ax.axhline(0, color="0.25", linewidth=0.7, zorder=0)
        ax.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
        ax.set_xlabel("Paired seed")
        ax.set_ylabel(r"$\Delta$ TEST " + ("corruption F1" if metric == "F1" else "BCE"))
        ax.set_ylim(data["plot"]["y_limits"])
        ax.set_yticks([-0.04, -0.03, -0.02, -0.01, 0.0])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlim(-0.25, len(seeds) - 0.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=0.6, length=3)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.22), borderaxespad=0, frameon=False,
                  handlelength=2.2, labelspacing=0.35)
        stem = OUTPUT / f"paired_{metric.lower()}_differences"
        fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
        fig.savefig(stem.with_suffix(".png"), dpi=data["plot"]["png_dpi"])
        plt.close(fig)
    print(f"Authenticated report-only figures written to {OUTPUT}")


if __name__ == "__main__":
    main()
