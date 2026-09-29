"""Create four publication figures from Phase 24 CSV result tables."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MASTER = ROOT / "results/tables/phase24_master_results.csv"
CONDITIONS = ROOT / "results/tables/phase24_condition_results.csv"
OUTPUT_DIR = ROOT / "results/figures/phase24"


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MASTER.is_file() or not CONDITIONS.is_file():
        raise FileNotFoundError("Generate Phase 24 result tables before the figures")
    master = pd.read_csv(MASTER)
    conditions = pd.read_csv(CONDITIONS)
    if master["evaluation_view"].tolist() != ["Baseline", "Gated - all records", "Gated - covered only"]:
        raise ValueError("Unexpected Phase 24 master result views")
    if len(conditions) != 6 or conditions["condition"].duplicated().any():
        raise ValueError("Phase 24 condition table must contain six unique conditions")
    return master, conditions


def _coverage_figure(master: pd.DataFrame, path: Path) -> None:
    gated = master[master["evaluation_view"] == "Gated - all records"].iloc[0]
    covered = int(round(gated["coverage"] * gated["N"]))
    abstained = int(round(gated["abstention_rate"] * gated["N"]))
    fig, ax = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
    ax.barh(["Eligible records"], [covered], label=f"Covered: {covered} ({gated['coverage'] * 100:.2f}%)")
    ax.barh(["Eligible records"], [abstained], left=[covered], label=f"Abstained: {abstained} ({gated['abstention_rate'] * 100:.2f}%)")
    ax.set_xlim(0, int(gated["N"]))
    ax.set_xlabel("Records (N)")
    ax.set_title(f"Gate coverage: {int(gated['N'])} eligible records")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=2, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _overall_quality_figure(master: pd.DataFrame, path: Path) -> None:
    views = master.set_index("evaluation_view").loc[
        ["Baseline", "Gated - all records", "Gated - covered only"]
    ]
    labels = [
        f"Baseline\nN={int(views.loc['Baseline', 'N'])}",
        f"Gated all\nN={int(views.loc['Gated - all records', 'N'])}",
        f"Gated covered\nN={int(views.loc['Gated - covered only', 'N'])}",
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), constrained_layout=True)
    for ax, metric, title in (
        (axes[0], "radgraph", "RadGraph-F1"),
        (axes[1], "chexpert", "CheXpert-F1"),
    ):
        bars = ax.bar(x, views[f"{metric}_mean"])
        for bar, value in zip(bars, views[f"{metric}_mean"]):
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9,
            )
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Mean report-level F1 by evaluation view")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _condition_figure(conditions: pd.DataFrame, path: Path) -> None:
    data = conditions.copy()
    labels = data["condition"].str.replace("_", "\n", regex=False).tolist()
    x = np.arange(len(data))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, metric, title in (
        (axes[0], "radgraph_f1", "RadGraph-F1 mean"),
        (axes[1], "chexpert_f1", "CheXpert-F1 mean"),
    ):
        ax.bar(x - width / 2, data[f"baseline_{metric}"], width, label="Baseline")
        ax.bar(x + width / 2, data[f"gated_{metric}"], width, label="Gated, all records")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Condition-level report metric means")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _unsupported_figure(master: pd.DataFrame, path: Path) -> None:
    views = master.set_index("evaluation_view").loc[
        ["Baseline", "Gated - all records", "Gated - covered only"]
    ]
    labels = ["Baseline", "Gated\nall records", "Gated\ncovered only"]
    x = np.arange(len(labels))
    rates = views["unsupported_claim_rate"].to_numpy(dtype=float) * 100
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    bars = ax.bar(x, rates)
    for bar, (_, row) in zip(bars, views.iterrows()):
        count = int(row["unsupported_claim_count"])
        denominator = int(row["generated_candidate_count"])
        ax.annotate(
            f"{count}/{denominator}\n{row['unsupported_claim_rate'] * 100:.2f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Unsupported claims / generated candidates (%)")
    ax.set_title("Unsupported-claim rate by report view")
    ax.set_ylim(0, max(rates) * 1.32 if len(rates) else 1)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures() -> list[Path]:
    master, conditions = _load()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        OUTPUT_DIR / "phase24_coverage_abstention.png",
        OUTPUT_DIR / "phase24_overall_quality.png",
        OUTPUT_DIR / "phase24_condition_quality.png",
        OUTPUT_DIR / "phase24_unsupported_claim_rates.png",
    ]
    _coverage_figure(master, outputs[0])
    _overall_quality_figure(master, outputs[1])
    _condition_figure(conditions, outputs[2])
    _unsupported_figure(master, outputs[3])
    return outputs


def main() -> None:
    files = create_figures()
    print("Created publication figures:")
    for path in files:
        print(path.relative_to(ROOT))
    print("No continuous risk-coverage curve was produced; the experiment has one deterministic gate operating point.")


if __name__ == "__main__":
    main()
