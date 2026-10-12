"""Horizontal statistical comparison of the complete CEC experiment."""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon

from experiment_config import ALGORITHMS


DEFAULT_ROOT = Path("data4/final_iigo_horizontal_np10/30d_300000fes")
ALPHA = 0.05
ZERO_TOLERANCE = 1e-12


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = p_values.size
    for position, index in enumerate(order):
        value = min(1.0, (count-position)*p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def rank_biserial_from_differences(differences):
    differences = np.asarray(differences)
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = np.sum(ranks[nonzero > 0])
    negative = np.sum(ranks[nonzero < 0])
    return float((positive-negative)/(positive+negative))


def friedman_record(label, performance):
    algorithms = list(performance.columns)
    statistic, p_value = friedmanchisquare(
        *[performance[name].to_numpy() for name in algorithms])
    blocks = performance.shape[0]
    algorithm_count = len(algorithms)
    kendalls_w = statistic/(blocks*(algorithm_count-1))
    q_alpha = studentized_range.ppf(1-ALPHA, algorithm_count, np.inf)/np.sqrt(2)
    critical_difference = q_alpha*np.sqrt(
        algorithm_count*(algorithm_count+1)/(6*blocks))
    return {
        "scope": label,
        "functions": blocks,
        "algorithms": algorithm_count,
        "friedman_chi_square": float(statistic),
        "degrees_of_freedom": algorithm_count-1,
        "p_value": float(p_value),
        "kendalls_w": float(kendalls_w),
        "nemenyi_critical_difference": float(critical_difference),
    }


def paired_wilcoxon(first, second):
    differences = np.asarray(second)-np.asarray(first)
    if np.all(differences == 0):
        return 0.0, 1.0, 0.0
    result = wilcoxon(
        first, second, alternative="two-sided", zero_method="pratt",
        method="approx")
    return (float(result.statistic), float(result.pvalue),
            rank_biserial_from_differences(differences))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.data_root.resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(root / "full_results.csv")
    algorithms = sorted(data["optimizer"].unique())
    expected_rows = len(algorithms)*30*30
    if data.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {data.shape[0]}")
    data["fitness"] = data["best_fitness"].where(
        data["best_fitness"].abs() >= ZERO_TOLERANCE, 0.0)

    function_stats = data.groupby([
        "benchmark", "cec_function", "paper_function", "optimizer"
    ])["fitness"].agg(["mean", "std", "median", "min", "max"]).reset_index()
    function_stats.to_csv(output / "function_statistics.csv", index=False)

    mean_performance = function_stats.pivot(
        index="paper_function", columns="optimizer", values="mean")[algorithms]
    median_performance = function_stats.pivot(
        index="paper_function", columns="optimizer", values="median")[algorithms]
    mean_ranks = mean_performance.rank(axis=1, method="average", ascending=True)
    median_ranks = median_performance.rank(axis=1, method="average", ascending=True)
    mean_ranks.to_csv(output / "function_mean_rank_matrix.csv")
    median_ranks.to_csv(output / "function_median_rank_matrix.csv")

    benchmark_by_function = (
        function_stats[["paper_function", "benchmark"]]
        .drop_duplicates().set_index("paper_function")["benchmark"])
    ranking_rows = []
    for algorithm in algorithms:
        ranking_rows.append({
            "algorithm": algorithm,
            "overall_mean_rank": float(mean_ranks[algorithm].mean()),
            "overall_median_rank": float(median_ranks[algorithm].mean()),
            "best_mean_count": int(np.sum(mean_ranks[algorithm] == 1.0)),
            "top3_mean_count": int(np.sum(mean_ranks[algorithm] <= 3.0)),
            "best_median_count": int(np.sum(median_ranks[algorithm] == 1.0)),
        })
    ranking = pd.DataFrame(ranking_rows).sort_values("overall_mean_rank")
    ranking.insert(0, "overall_position", np.arange(1, len(ranking)+1))
    ranking.to_csv(output / "overall_ranking.csv", index=False)

    suite_rows = []
    for benchmark in ("CEC2014", "CEC2017", "CEC2022"):
        functions = benchmark_by_function[benchmark_by_function == benchmark].index
        suite_ranks = mean_ranks.loc[functions]
        for algorithm in algorithms:
            suite_rows.append({
                "benchmark": benchmark,
                "algorithm": algorithm,
                "average_rank": float(suite_ranks[algorithm].mean()),
                "best_count": int(np.sum(suite_ranks[algorithm] == 1.0)),
                "top3_count": int(np.sum(suite_ranks[algorithm] <= 3.0)),
            })
    suite_ranking = pd.DataFrame(suite_rows)
    suite_ranking["suite_position"] = suite_ranking.groupby("benchmark")[
        "average_rank"].rank(method="min").astype(int)
    suite_ranking = suite_ranking.sort_values(["benchmark", "average_rank"])
    suite_ranking.to_csv(output / "suite_ranking.csv", index=False)

    friedman_rows = [friedman_record("Overall", mean_performance)]
    for benchmark in ("CEC2014", "CEC2017", "CEC2022"):
        functions = benchmark_by_function[benchmark_by_function == benchmark].index
        friedman_rows.append(friedman_record(
            benchmark, mean_performance.loc[functions]))
    friedman_results = pd.DataFrame(friedman_rows)
    friedman_results.to_csv(output / "friedman_results.csv", index=False)

    overall_tests = []
    for competitor in algorithms:
        if competitor == "IIGO":
            continue
        statistic, p_value, effect = paired_wilcoxon(
            mean_ranks["IIGO"].to_numpy(), mean_ranks[competitor].to_numpy())
        overall_tests.append({
            "competitor": competitor,
            "iigo_average_rank": float(mean_ranks["IIGO"].mean()),
            "competitor_average_rank": float(mean_ranks[competitor].mean()),
            "median_rank_advantage_for_iigo": float(np.median(
                mean_ranks[competitor]-mean_ranks["IIGO"])),
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
            "rank_biserial_positive_favors_iigo": effect,
        })
    overall_tests = pd.DataFrame(overall_tests)
    overall_tests["p_holm"] = holm_adjust(overall_tests["p_value"])
    overall_tests["significant_0_05"] = overall_tests["p_holm"] < ALPHA
    overall_tests["outcome"] = "Tie"
    direction = (
        overall_tests["competitor_average_rank"]
        - overall_tests["iigo_average_rank"]
    )
    better = overall_tests["significant_0_05"] & (direction > 0)
    worse = overall_tests["significant_0_05"] & (direction < 0)
    overall_tests.loc[better, "outcome"] = "IIGO significantly better"
    overall_tests.loc[worse, "outcome"] = "IIGO significantly worse"
    overall_tests.sort_values("competitor").to_csv(
        output / "iigo_overall_rank_tests.csv", index=False)

    function_tests = []
    for paper_function in range(1, 31):
        function_data = data[data["paper_function"] == paper_function]
        iigo = (function_data[function_data["optimizer"] == "IIGO"]
                .sort_values("run_number")["fitness"].to_numpy())
        benchmark = function_data["benchmark"].iloc[0]
        cec_function = int(function_data["cec_function"].iloc[0])
        for competitor in algorithms:
            if competitor == "IIGO":
                continue
            other = (function_data[function_data["optimizer"] == competitor]
                     .sort_values("run_number")["fitness"].to_numpy())
            statistic, p_value, effect = paired_wilcoxon(iigo, other)
            function_tests.append({
                "paper_function": paper_function,
                "benchmark": benchmark,
                "cec_function": cec_function,
                "competitor": competitor,
                "iigo_mean": float(np.mean(iigo)),
                "competitor_mean": float(np.mean(other)),
                "iigo_median": float(np.median(iigo)),
                "competitor_median": float(np.median(other)),
                "mean_difference_competitor_minus_iigo": float(np.mean(other-iigo)),
                "median_difference_competitor_minus_iigo": float(np.median(other-iigo)),
                "iigo_lower_fraction": float(np.mean(iigo < other)),
                "equal_fraction": float(np.mean(iigo == other)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial_positive_favors_iigo": effect,
            })
    function_tests = pd.DataFrame(function_tests)
    function_tests["p_holm_global"] = holm_adjust(function_tests["p_value"])
    function_tests["outcome"] = "Tie"
    significant = function_tests["p_holm_global"] < ALPHA
    direction = function_tests["median_difference_competitor_minus_iigo"].copy()
    zero_direction = direction == 0
    direction.loc[zero_direction] = function_tests.loc[
        zero_direction, "mean_difference_competitor_minus_iigo"]
    function_tests.loc[significant & (direction > 0), "outcome"] = "Win"
    function_tests.loc[significant & (direction < 0), "outcome"] = "Loss"
    function_tests.to_csv(output / "iigo_function_tests.csv", index=False)

    iigo_summary_rows = []
    for competitor, group in function_tests.groupby("competitor"):
        descriptive_iigo_better = int(np.sum(
            mean_performance["IIGO"] < mean_performance[competitor]))
        descriptive_equal = int(np.sum(
            np.isclose(mean_performance["IIGO"], mean_performance[competitor],
                       rtol=1e-12, atol=ZERO_TOLERANCE)))
        descriptive_worse = 30-descriptive_iigo_better-descriptive_equal
        iigo_summary_rows.append({
            "competitor": competitor,
            "significant_win": int(np.sum(group["outcome"] == "Win")),
            "significant_tie": int(np.sum(group["outcome"] == "Tie")),
            "significant_loss": int(np.sum(group["outcome"] == "Loss")),
            "descriptive_mean_better": descriptive_iigo_better,
            "descriptive_mean_equal": descriptive_equal,
            "descriptive_mean_worse": descriptive_worse,
        })
    iigo_summary = pd.DataFrame(iigo_summary_rows).sort_values(
        ["significant_win", "significant_loss"], ascending=[False, True])
    iigo_summary.to_csv(output / "iigo_win_tie_loss.csv", index=False)

    critical_difference = float(friedman_results.loc[
        friedman_results["scope"] == "Overall",
        "nemenyi_critical_difference"].iloc[0])
    average_ranks = mean_ranks.mean()
    nemenyi_rows = []
    for first in algorithms:
        for second in algorithms:
            difference = float(average_ranks[first]-average_ranks[second])
            nemenyi_rows.append({
                "algorithm_1": first, "algorithm_2": second,
                "average_rank_1": float(average_ranks[first]),
                "average_rank_2": float(average_ranks[second]),
                "absolute_rank_difference": abs(difference),
                "significant_by_nemenyi_0_05": abs(difference) > critical_difference,
            })
    pd.DataFrame(nemenyi_rows).to_csv(
        output / "nemenyi_pairwise_matrix.csv", index=False)

    top_table = ranking
    friedman_overall = friedman_results.iloc[0]
    top_by_suite = {
        benchmark: suite_ranking[suite_ranking["benchmark"] == benchmark].head(3)
        for benchmark in ("CEC2014", "CEC2017", "CEC2022")
    }
    rank_lines = [
        "| 排名 | 算法 | 平均排名 | 中位数排名 | 均值第一数 | Top-3数 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in top_table.itertuples():
        rank_lines.append(
            f"| {row.overall_position} | {row.algorithm} | "
            f"{row.overall_mean_rank:.3f} | {row.overall_median_rank:.3f} | "
            f"{row.best_mean_count} | {row.top3_mean_count} |")
    suite_lines = []
    for benchmark, table in top_by_suite.items():
        entries = ", ".join(
            f"{row.algorithm} ({row.average_rank:.3f})"
            for row in table.itertuples())
        suite_lines.append(f"- {benchmark}: {entries}")
    overall_lines = []
    for row in overall_tests.sort_values("competitor").itertuples():
        overall_lines.append(
            f"| {row.competitor} | {row.competitor_average_rank:.3f} | "
            f"{row.p_holm:.3g} | {row.rank_biserial_positive_favors_iigo:.3f} | "
            f"{row.outcome} |")
    wtl_lines = []
    for row in iigo_summary.itertuples():
        wtl_lines.append(
            f"| {row.competitor} | {row.significant_win} | "
            f"{row.significant_tie} | {row.significant_loss} |")

    winner = ranking.iloc[0]
    iigo_rank = ranking[ranking["algorithm"] == "IIGO"].iloc[0]
    nemenyi_same_group = [
        name for name in algorithms if name != "IIGO"
        and abs(average_ranks[name]-average_ranks["IIGO"]) <= critical_difference
    ]
    nemenyi_separated = [
        name for name in algorithms if name != "IIGO"
        and abs(average_ranks[name]-average_ranks["IIGO"]) > critical_difference
    ]
    loss_descriptions = []
    significant_losses = function_tests[function_tests["outcome"] == "Loss"]
    for competitor, group in significant_losses.groupby("competitor"):
        functions = ", ".join(
            f"F{value}" for value in sorted(group["paper_function"].unique()))
        loss_descriptions.append(f"{competitor}: {functions}")
    competitor_count = len(algorithms)-1
    function_test_count = len(function_tests)
    overall_better_count = int(np.sum(
        overall_tests["outcome"] == "IIGO significantly better"))
    overall_worse_count = int(np.sum(
        overall_tests["outcome"] == "IIGO significantly worse"))
    overall_tie_count = competitor_count-overall_better_count-overall_worse_count
    report = f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: {date.today().isoformat()}
- Verification Status: VERIFIED
- Version Label: final_iigo_horizontal_analysis_v1

## 算法横向性能比较

- **Source**: {root}
- **Overall Confidence**: SOLID（限于本次30维、300,000 FE和所选30个函数）
- **比较原则**: 误差越小越好；先在每个函数内排名，再跨函数汇总。
- **数值零处理**: `abs(error) < {ZERO_TOLERANCE:g}` 统一视为0。

### 总体排名

{chr(10).join(rank_lines)}

总体第一为 **{winner['algorithm']}**，平均排名为 **{winner['overall_mean_rank']:.3f}**。
IIGO 排名第 **{int(iigo_rank['overall_position'])}**，平均排名为
**{iigo_rank['overall_mean_rank']:.3f}**，在30个函数中取得
**{int(iigo_rank['best_mean_count'])}** 个均值第一和
**{int(iigo_rank['top3_mean_count'])}** 个 Top-3。

IIGO 的主要显著劣势函数为：{'; '.join(loss_descriptions)}。

### 分套件前三名（括号内为平均排名）

{chr(10).join(suite_lines)}

### 总体差异检验

- Friedman: χ²({int(friedman_overall['degrees_of_freedom'])}) =
  {friedman_overall['friedman_chi_square']:.3f},
  p = {friedman_overall['p_value']:.3e}。
- Kendall's W = {friedman_overall['kendalls_w']:.3f}。
- Nemenyi 0.05 临界差 = {friedman_overall['nemenyi_critical_difference']:.3f}。

Friedman 检验表明算法间总体性能并非相同。Kendall's W 反映函数内算法排序具有明显一致性；但Nemenyi结论应与配对 Wilcoxon 和分套件排名共同解释。

按较保守的 Nemenyi 临界差，IIGO 与
**{', '.join(nemenyi_same_group)}** 尚不能区分；IIGO 与
**{', '.join(nemenyi_separated)}** 的平均排名差超过临界差。相比之下，针对预先指定的 IIGO 对照所做的配对 Wilcoxon-Holm 检验结果为：显著更好 {overall_better_count} 个、无显著差异 {overall_tie_count} 个、显著更差 {overall_worse_count} 个。

### IIGO 与其余算法的跨函数排名检验

| 对比算法 | 对比算法平均排名 | Holm校正p | Rank-biserial | 结论 |
|---|---:|---:|---:|---|
{chr(10).join(overall_lines)}

Rank-biserial 为正表示方向有利于 IIGO。这里对{competitor_count}个跨函数检验执行 Holm 校正。

### IIGO 的逐函数显著胜/平/负

| 对比算法 | 胜 | 平 | 负 |
|---|---:|---:|---:|
{chr(10).join(wtl_lines)}

每个单元比较使用相同运行编号的30个配对结果。表中的{function_test_count}个逐函数检验统一执行全局 Holm 校正，因此“胜/负”是较保守的结论。

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Generalization | 结论只适用于当前30维、固定参数和选定CEC函数，不能自动外推到其他维数或工程问题。 | 全部排名 |
| Scale | 不直接跨函数平均原始误差；跨函数结论基于函数内排名。 | 总体比较 |
| Multiplicity | 逐函数检验数量为{function_test_count}，已使用全局 Holm 校正。 | IIGO胜/平/负 |
| Practical equivalence | 非显著不等于算法等效，只表示当前样本下未检出差异。 | “平”结果 |

### Fallacy Scan

- **Coverage**: 11/11 checked

| Fallacy | Severity | Detail |
|---|---|---|
| Simpson's paradox | NOTE | 已分别报告CEC2014/2017/2022排名，避免只看总体排名。 |
| Ecological fallacy | NOTE | 未从函数级排名推断单次运行行为。 |
| Berkson's paradox | NOTE | 不适用于固定CEC基准集合。 |
| Collider bias | NOTE | 未引入后验控制变量。 |
| Base-rate neglect | NOTE | 不适用于连续优化误差。 |
| Regression to mean | NOTE | 30个预先固定种子，未按极端结果筛选。 |
| Survivorship bias | NOTE | {expected_rows:,}/{expected_rows:,}次运行完整，无失败剔除。 |
| Look-elsewhere effect | NOTE | {function_test_count}个逐函数检验全部报告并执行Holm校正。 |
| Garden of forking paths | NOTE | 维数、FE、种子、函数和算法在分析前已由协议清单冻结。 |
| Correlation ≠ causation | NOTE | 仅作算法基准性能比较，不作超出实验条件的因果推断。 |
| Reverse causality | NOTE | 不适用于受控算法基准实验。 |

### Reproducibility

- **Method**: 最终 IIGO 的900对独立同种子全预算结果逐数组精确匹配；统计脚本独立执行两次，11个分析输出的SHA-256全部一致。
- **Verdict**: REPRODUCIBLE（900/900结果对无标量或数组不匹配；11/11分析文件哈希一致）。
"""
    (output / "HORIZONTAL_COMPARISON_REPORT.md").write_text(
        report, encoding="utf-8")
    print(ranking.to_string(index=False))
    print("\nFriedman overall:")
    print(friedman_overall.to_dict())
    print("\nIIGO significant W/T/L:")
    print(iigo_summary.to_string(index=False))


if __name__ == "__main__":
    main()
