"""Compare strict verification reports across experiment runs."""

import json
import os

import numpy as np

from analysis.verify import verify


def summarize_reports(reports):
    """Return mean and standard deviation for core run-level metrics."""
    if not reports:
        raise ValueError("At least one report is required.")
    metrics = {
        "normal_efficiency": [
            report["results"]["normal"]["efficiency"] for report in reports
        ],
        "zero_efficiency": [
            report["results"]["zero"]["efficiency"] for report in reports
        ],
        "random_efficiency": [
            report["results"]["random"]["efficiency"] for report in reports
        ],
        "minimum_language_advantage": [
            report["minimum_language_advantage"] for report in reports
        ],
    }
    return {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for name, values in metrics.items()
    }


def compare_runs(run_dirs, output_dir="figures/comparison", n_episodes=1000):
    """Verify several run directories and write an aggregate summary."""
    os.makedirs(output_dir, exist_ok=True)
    reports = []
    for run_dir in run_dirs:
        run_name = os.path.basename(os.path.normpath(run_dir))
        figure_dir = os.path.join(output_dir, run_name)
        reports.append(verify(run_dir, figure_dir, n_episodes))

    summary = {
        "runs": run_dirs,
        "n_episodes_per_control": n_episodes,
        "metrics": summarize_reports(reports),
    }
    path = os.path.join(output_dir, "comparison_summary.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print("\n=== MULTI-RUN SUMMARY ===")
    for name, values in summary["metrics"].items():
        print(f"{name:>28}: {values['mean']:.3f} +/- {values['std']:.3f}")
    print(f"Saved -> {path}")
    return summary
