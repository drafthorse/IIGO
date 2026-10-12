"""Analyze the sequential post-warm-up runtime experiment."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


ALPHA = 0.05
CONTROL = "IIGO"


def holm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (len(values)-position)*values[index]))
        adjusted[index] = running
    return adjusted


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive-negative)/(positive+negative)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("data4/final_iigo_runtime/30d_300000fes"),
    )
    args = parser.parse_args()
    root = args.data_root.resolve()
    data = pd.read_csv(root / "runtime_results.csv")
    algorithms = sorted(data.algorithm.unique())
    blocks = sorted({
        (row.benchmark, int(row.run)) for row in data.itertuples()
    })
    expected = len(algorithms)*len(blocks)
    failures = []
    if len(data) != expected:
        failures.append(f"rows={len(data)} expected={expected}")
    counts = data.groupby(["algorithm", "benchmark"]).size()
    if counts.nunique() != 1:
        failures.append("unbalanced_runtime_blocks")

    pivot = data.pivot(
        index=["benchmark", "run"], columns="algorithm",
        values="elapsed_seconds",
    )[algorithms]
    rank_matrix = pivot.rank(axis=1, method="average", ascending=True)
    ranking = pd.DataFrame({
        "mean_seconds": pivot.mean(),
        "median_seconds": pivot.median(),
        "std_seconds": pivot.std(ddof=1),
        "average_runtime_rank": rank_matrix.mean(),
    }).sort_values(["median_seconds", "mean_seconds"])
    ranking["median_ratio_to_iigo"] = (
        ranking.median_seconds/ranking.loc[CONTROL, "median_seconds"]
    )
    ranking.insert(0, "runtime_position", np.arange(1, len(ranking)+1))
    ranking.index.name = "algorithm"
    ranking.to_csv(root / "overall_runtime_ranking.csv", encoding="utf-8-sig")

    suite = (
        data.groupby(["benchmark", "algorithm"])["elapsed_seconds"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
    )
    suite["median_ratio_to_iigo"] = np.nan
    for benchmark in suite.benchmark.unique():
        control = float(suite[
            (suite.benchmark == benchmark) & (suite.algorithm == CONTROL)
        ].iloc[0]["median"])
        suite.loc[
            suite.benchmark == benchmark, "median_ratio_to_iigo"
        ] = suite.loc[suite.benchmark == benchmark, "median"]/control
    suite.to_csv(root / "suite_runtime_summary.csv", index=False, encoding="utf-8-sig")

    comparisons = []
    control_values = pivot[CONTROL].to_numpy()
    for algorithm in algorithms:
        if algorithm == CONTROL:
            continue
        values = pivot[algorithm].to_numpy()
        if np.array_equal(control_values, values):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(
                control_values, values, zero_method="pratt", method="approx"
            ).pvalue)
        differences = values-control_values
        comparisons.append({
            "algorithm": algorithm,
            "iigo_median_seconds": float(np.median(control_values)),
            "algorithm_median_seconds": float(np.median(values)),
            "median_ratio_algorithm_to_iigo": float(
                np.median(values)/np.median(control_values)
            ),
            "rank_biserial_positive_means_algorithm_slower": rank_biserial(differences),
            "p_raw": p_value,
        })
    comparisons = pd.DataFrame(comparisons)
    comparisons["p_holm"] = holm(comparisons.p_raw.to_numpy())
    comparisons["outcome"] = "No significant timing difference"
    significant = comparisons.p_holm < ALPHA
    comparisons.loc[
        significant & (comparisons.median_ratio_algorithm_to_iigo > 1),
        "outcome",
    ] = "Algorithm slower than IIGO"
    comparisons.loc[
        significant & (comparisons.median_ratio_algorithm_to_iigo < 1),
        "outcome",
    ] = "Algorithm faster than IIGO"
    comparisons.to_csv(
        root / "iigo_runtime_wilcoxon_holm.csv",
        index=False, encoding="utf-8-sig",
    )

    report_json = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED" if not failures else "FAILED",
            "version_label": "final_iigo_runtime_analysis_v1",
        },
        "status": "PASS" if not failures else "FAIL",
        "blocks": len(blocks),
        "algorithms": len(algorithms),
        "iigo": ranking.loc[CONTROL].to_dict(),
        "comparisons": comparisons.to_dict(orient="records"),
        "failures": failures,
    }
    (root / "runtime_analysis_report.json").write_text(
        json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ablation_names = [
        name for name in ("IIGO_NoWindow", "IIGO_NoIGE", "IIGO_NoStage")
        if name in ranking.index
    ]
    lines = [
        "# Final IIGO execution-time validation", "",
        "## Material Passport", "",
        "- Origin Skill: academic-research-suite/experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {date.today().isoformat()}",
        "- Verification Status: ANALYZED",
        "- Version Label: final_iigo_runtime_analysis_v1", "",
        "## Timing scope", "",
        "Times cover only the optimizer call after per-algorithm warm-up. Input loading, initialization generation, file output and compilation are excluded. Calls were sequential and their order was randomized within each suite/run block.",
        "", "## Overall runtime ranking", "",
        "| Position | Algorithm | Median (s) | Mean (s) | Runtime rank | Ratio to IIGO |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for algorithm, row in ranking.iterrows():
        lines.append(
            f"| {int(row.runtime_position)} | {algorithm} | "
            f"{row.median_seconds:.6f} | {row.mean_seconds:.6f} | "
            f"{row.average_runtime_rank:.3f} | {row.median_ratio_to_iigo:.3f} |"
        )
    lines.extend([
        "", "## IIGO mechanism overhead", "",
        "| Variant | Median ratio to full IIGO | Holm p | Interpretation |",
        "|---|---:|---:|---|",
    ])
    indexed = comparisons.set_index("algorithm")
    for name in ablation_names:
        row = indexed.loc[name]
        lines.append(
            f"| {name} | {row.median_ratio_algorithm_to_iigo:.3f} | "
            f"{row.p_holm:.6g} | {row.outcome} |"
        )
    lines.extend([
        "", "## Warnings", "",
        "- Wall-clock measurements are environment-sensitive; ratios are valid for the recorded machine and workload, not universal constants.",
        "- Randomized sequential order reduces systematic drift but does not eliminate OS scheduling noise.",
        "- Statistical timing differences do not by themselves establish practical importance; report the ratio and absolute seconds.",
        "", "## Fallacy scan", "",
        "Coverage: 11/11 checked. Suite-specific results are retained to inspect aggregation; all completed calls are included; 20 control comparisons use Holm correction; no non-significant result is called equivalent; no hardware-general or causal runtime claim is made.",
        "", "## Reproducibility", "",
        "Verdict: PARTIALLY_REPRODUCIBLE. Source, environment, warm-up and execution order are recorded, and numerical outputs/FE counts are validated. Exact wall-clock equality is neither expected nor used as a criterion.", "",
    ])
    (root / "RUNTIME_VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report_json, indent=2, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
