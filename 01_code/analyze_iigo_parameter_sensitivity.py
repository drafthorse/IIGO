"""Analyze IIGO sensitivity experiments and assemble one manuscript table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parent
PARAMETER_ROOT = ROOT / "data4" / "iigo_final_parameter_sensitivity" / "full_30d_300000fes"
POPULATION_ROOT = ROOT / "data4" / "iigo_final_population_sensitivity" / "full_r30_300000fes"

GROUPS = {
    "gr": [
        ("gr_0.05", "0.05"), ("gr_0.10", "0.10"),
        ("baseline", "0.15"), ("gr_0.20", "0.20"), ("gr_0.25", "0.25"),
    ],
    "Window fraction": [
        ("window_0.10", "0.10 (2)"), ("window_0.20", "0.20 (3)"),
        ("baseline", "0.30 (4)"), ("window_0.40", "0.40 (5)"),
        ("window_0.50", "0.50 (6)"),
    ],
    "IGE modulation": [
        ("baseline", "Cosine"), ("modulation_gaussian", "Gaussian"),
    ],
}


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - position) * values[index]))
        adjusted[index] = running
    return adjusted


def signed_rank(left, right):
    if np.array_equal(left, right):
        return 1.0
    return float(wilcoxon(left, right, zero_method="pratt", method="approx").pvalue)


def rows_from_matrix(parameter, means, labels, default):
    ranks = means.rank(axis=1, method="average", ascending=True)
    others = [column for column in means.columns if column != default]
    raw = [signed_rank(ranks[default].to_numpy(), ranks[item].to_numpy()) for item in others]
    adjusted = holm(raw)
    adjusted_map = dict(zip(others, adjusted))
    rows = []
    for condition in means.columns:
        candidate = means[condition]
        reference = means[default]
        if condition == default:
            wtl, p_value = "Reference", None
        else:
            wins = int((reference < candidate).sum())
            ties = int((candidate == reference).sum())
            losses = int((reference > candidate).sum())
            wtl = f"{wins}/{ties}/{losses}"
            p_value = float(adjusted_map[condition])
        rows.append({
            "Parameter": parameter,
            "Setting": labels[condition],
            "Default": "Yes" if condition == default else "No",
            "Average rank": float(ranks[condition].mean()),
            "First place count": int((ranks[condition] == 1).sum()),
            "Top three count": int((ranks[condition] <= 3).sum()),
            "W/T/L vs default": wtl,
            "Holm adjusted p": p_value,
        })
    if len(means.columns) > 2:
        test = friedmanchisquare(*(means[column].to_numpy() for column in means.columns))
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
        kendalls_w = statistic / (len(means) * (len(means.columns) - 1))
    else:
        statistic = p_value = kendalls_w = None
    statistics = {
        "parameter": parameter,
        "friedman_statistic": statistic,
        "friedman_p": p_value,
        "kendalls_w": kendalls_w,
        "best_setting": labels[ranks.mean().idxmin()],
        "default_setting": labels[default],
    }
    return rows, statistics


def summarize_parameter_group(data, parameter, members):
    order = [condition for condition, _ in members]
    labels = dict(members)
    means = (
        data[data["condition"].isin(order)]
        .groupby(["paper_function", "condition"])["best_fitness"]
        .mean().unstack("condition").reindex(columns=order)
    )
    return rows_from_matrix(parameter, means, labels, "baseline")


def summarize_population(data):
    populations = [10, 20, 30, 40, 50]
    means = (
        data[data["population_size"].isin(populations)]
        .groupby(["paper_function", "population_size"])["best_fitness"]
        .mean().unstack("population_size").reindex(columns=populations)
    )
    labels = {population: str(population) for population in populations}
    return rows_from_matrix("Initial population size", means, labels, 10)


def main():
    parameter_data = pd.read_csv(PARAMETER_ROOT / "full_results.csv")
    population_data = pd.read_csv(POPULATION_ROOT / "full_results.csv")
    failures = []
    if len(parameter_data) != 9000:
        failures.append(f"parameter_rows={len(parameter_data)}")
    if parameter_data["paper_function"].nunique() != 30:
        failures.append("parameter_function_count")
    counts = parameter_data.groupby(["condition", "paper_function"]).size()
    if not (counts == 30).all():
        failures.append("parameter_run_count")

    all_rows, all_statistics = [], []
    rows, statistics = summarize_population(population_data)
    all_rows.extend(rows)
    all_statistics.append(statistics)
    for parameter, members in GROUPS.items():
        rows, statistics = summarize_parameter_group(parameter_data, parameter, members)
        all_rows.extend(rows)
        all_statistics.append(statistics)

    table = pd.DataFrame(all_rows)
    table.to_csv(
        PARAMETER_ROOT / "manuscript_parameter_sensitivity_table.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(all_statistics).to_csv(
        PARAMETER_ROOT / "parameter_group_statistics.csv",
        index=False, encoding="utf-8-sig",
    )
    report = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "verification_status": "VERIFIED" if not failures else "FAILED",
            "version_label": "iigo_parameter_sensitivity_analysis_v1",
        },
        "status": "PASS" if not failures else "FAIL",
        "protocol": {
            "dimension": 30,
            "population_size_for_non_population_tests": 10,
            "functions": 30,
            "runs": 30,
            "max_fes": 300000,
            "ranking_basis": "mean final fitness over 30 runs per function",
            "statistics": (
                "Friedman test within each multilevel factor; paired Wilcoxon "
                "signed rank tests on function ranks versus the default setting "
                "with Holm correction."
            ),
        },
        "group_statistics": all_statistics,
        "rows": all_rows,
        "failures": failures,
    }
    (PARAMETER_ROOT / "analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
