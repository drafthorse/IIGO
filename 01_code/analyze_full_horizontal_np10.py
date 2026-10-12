"""Horizontal comparison of the full NP_init=10 CEC experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "data4/final_iigo_horizontal_np10/30d_300000fes/full_summary.csv"


def function_category(suite: str, cec_function: int) -> str:
    if suite == "CEC2014":
        return ("unimodal" if cec_function <= 3 else "multimodal"
                if cec_function <= 16 else "hybrid"
                if cec_function <= 22 else "composition")
    if suite == "CEC2017":
        return ("unimodal" if cec_function <= 3 else "multimodal"
                if cec_function <= 10 else "hybrid"
                if cec_function <= 20 else "composition")
    return "basic" if cec_function <= 5 else "hybrid" if cec_function <= 8 else "composition"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    algorithms = sorted({row["optimizer"] for row in rows})
    functions = sorted(
        {(row["benchmark"], int(row["paper_function"])) for row in rows},
        key=lambda item: item[1],
    )
    by_key = {
        (row["optimizer"], row["benchmark"], int(row["paper_function"])): row
        for row in rows
    }
    means = np.asarray([
        [float(by_key[(algorithm, suite, function)]["mean"])
         for algorithm in algorithms]
        for suite, function in functions
    ])
    stds = np.asarray([
        [float(by_key[(algorithm, suite, function)]["std"])
         for algorithm in algorithms]
        for suite, function in functions
    ])
    mean_ranks = np.vstack([rankdata(values, method="average") for values in means])
    std_ranks = np.vstack([rankdata(values, method="average") for values in stds])

    suite_ranks = {}
    for suite in sorted({suite for suite, _ in functions}):
        indices = [i for i, (name, _) in enumerate(functions) if name == suite]
        suite_ranks[suite] = mean_ranks[indices].mean(axis=0)

    iigo = algorithms.index("IIGO")
    ranking = []
    for index, algorithm in enumerate(algorithms):
        comparison = []
        for row_index in range(len(functions)):
            left = means[row_index, iigo]
            right = means[row_index, index]
            tolerance = 1e-12 * max(1.0, abs(left), abs(right))
            comparison.append(-1 if left < right - tolerance else
                              1 if left > right + tolerance else 0)
        ranking.append({
            "algorithm": algorithm,
            "mean_rank": float(mean_ranks[:, index].mean()),
            "best_or_numerically_tied_count": int(sum(
                means[row_index, index] <= np.min(means[row_index]) +
                1e-12 * max(1.0, abs(float(np.min(means[row_index]))))
                for row_index in range(len(functions))
            )),
            "top3_count": int(np.sum(mean_ranks[:, index] <= 3.0)),
            "stability_rank": float(std_ranks[:, index].mean()),
            "cec2014_rank": float(suite_ranks["CEC2014"][index]),
            "cec2017_rank": float(suite_ranks["CEC2017"][index]),
            "cec2022_rank": float(suite_ranks["CEC2022"][index]),
            "iigo_vs_algorithm_w_t_l": [
                int(comparison.count(-1)),
                int(comparison.count(0)),
                int(comparison.count(1)),
            ],
        })
    ranking.sort(key=lambda item: (item["mean_rank"], item["stability_rank"]))
    for position, item in enumerate(ranking, start=1):
        item["overall_position"] = position

    statistic, p_value = friedmanchisquare(*[means[:, i] for i in range(len(algorithms))])
    iigo_nonwins = []
    for row_index, (suite, function) in enumerate(functions):
        best = float(np.min(means[row_index]))
        tolerance = 1e-12 * max(1.0, abs(best))
        winners = [
            algorithm for index, algorithm in enumerate(algorithms)
            if means[row_index, index] <= best + tolerance
        ]
        if "IIGO" not in winners:
            iigo_nonwins.append({
                "function": function,
                "suite": suite,
                "iigo_rank": float(mean_ranks[row_index, iigo]),
                "winner": winners,
                "iigo_mean": float(means[row_index, iigo]),
                "best_mean": best,
            })
    categories: dict[str, list[float]] = {}
    for row_index, (suite, function) in enumerate(functions):
        cec_function = int(by_key[("IIGO", suite, function)]["cec_function"])
        categories.setdefault(function_category(suite, cec_function), []).append(
            float(mean_ranks[row_index, iigo])
        )
    report = {
        "metric": "30-run mean objective; lower is better",
        "source": str(args.summary.resolve()),
        "algorithms": len(algorithms),
        "functions": len(functions),
        "friedman_chi_square": float(statistic),
        "friedman_p_value": float(p_value),
        "ranking": ranking,
        "iigo_category_ranks": {
            name: {"average_rank": float(np.mean(values)), "functions": len(values)}
            for name, values in sorted(categories.items())
        },
        "iigo_nonwinning_functions": iigo_nonwins,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
