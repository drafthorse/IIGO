"""Run the D=100 CEC 2017 scalability experiment.

Protocol
--------
CEC 2017 functions F1 and F3--F30 (the official 29-function set), D=100,
NP_init=10, 1,000,000 actual objective evaluations, and 30 paired runs.
IIGO is compared with CSO, LSHADE, LSHADE-cnEpSin, jSO, and CMA-ES.

The runner is resumable.  Each completed run is saved atomically in an
isolated output directory and is validated before it is counted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Avoid nested BLAS/OpenMP parallelism inside the process pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numba
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

import Interface
from ExperimentAlgorithms import get_f
from elements import F
from experiment_config import BENCHMARK_IDS, RUN_SEEDS, derive_seed


ROOT = Path(__file__).resolve().parent
BENCHMARK = "CEC2017"
BENCHMARK_ID = BENCHMARK_IDS[BENCHMARK]
DIMENSION = 100
POPULATION_SIZE = 10
MAX_FES = 1_000_000
RUNS = 30
FUNCTIONS = (1,) + tuple(range(3, 31))
ALGORITHMS = (
    "IIGO",
    "CSO",
    "LSHADE",
    "LSHADE_cnEpSin",
    "jSO",
    "CMA-ES",
)
DEFAULT_OUTPUT = (
    ROOT / "data4" / "high_dimensional_scalability"
    / "cec2017_100d_np10_1000000fes"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_paths(output_root: Path, algorithm: str, function_number: int,
                 run: int) -> tuple[Path, Path]:
    folder = output_root / algorithm / f"F{function_number}"
    stem = f"{algorithm}_F{function_number}_{run}"
    return folder / f"{stem}.json", folder / f"{stem}.npz"


def metadata_complete(output_root: Path, algorithm: str,
                      function_number: int, run: int, max_fes: int) -> bool:
    metadata_path, data_path = result_paths(
        output_root, algorithm, function_number, run
    )
    if not metadata_path.exists() or not data_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("algorithm") != algorithm:
            return False
        if int(metadata.get("function", -1)) != function_number:
            return False
        if int(metadata.get("run", -1)) != run:
            return False
        if int(metadata.get("dimension", -1)) != DIMENSION:
            return False
        if int(metadata.get("population_size", -1)) != POPULATION_SIZE:
            return False
        if int(metadata.get("max_fes", -1)) != max_fes:
            return False
        if int(metadata.get("final_fes", -1)) != max_fes:
            return False
        if int(metadata.get("seed", -1)) != derive_seed(
            BENCHMARK, function_number, run
        ):
            return False
        with np.load(data_path) as payload:
            position = np.asarray(payload["best_position"])
            best = float(np.asarray(payload["best_fitness"]).reshape(-1)[0])
        return (
            position.shape == (DIMENSION,)
            and np.isfinite(position).all()
            and np.isfinite(best)
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_one(output_root_text: str, algorithm: str, function_number: int,
            run: int, max_fes: int) -> dict:
    output_root = Path(output_root_text)
    matrix, shift, shuffle = Interface.load_cec_input(
        BENCHMARK, function_number, DIMENSION
    )
    seed, initial_population, initial_hash = Interface.make_initial_population(
        BENCHMARK, function_number, run, POPULATION_SIZE, DIMENSION
    )
    started = time.perf_counter()
    convergence, evaluation_axis, best_position, best_fitness = get_f(
        algorithm
    )(
        max_fes,
        POPULATION_SIZE,
        -100.0,
        100.0,
        DIMENSION,
        matrix,
        shift,
        function_number,
        shuffle,
        BENCHMARK_ID,
        seed,
        initial_population,
    )
    elapsed = time.perf_counter() - started

    convergence = np.asarray(convergence, dtype=np.float64)
    evaluation_axis = np.asarray(evaluation_axis, dtype=np.int64)
    best_position = np.asarray(best_position, dtype=np.float64)
    best_fitness = float(best_fitness)
    if evaluation_axis.size == 0 or int(evaluation_axis[-1]) != max_fes:
        raise RuntimeError(
            f"budget mismatch {algorithm}/F{function_number}/run{run}"
        )
    if convergence.shape != evaluation_axis.shape:
        raise RuntimeError(
            f"curve mismatch {algorithm}/F{function_number}/run{run}"
        )
    if convergence.size > 1:
        tolerance = 1e-10 * np.maximum(1.0, np.abs(convergence[:-1]))
        if np.any(np.diff(convergence) > tolerance):
            raise RuntimeError(
                f"nonmonotonic curve {algorithm}/F{function_number}/run{run}"
            )
    if not np.isfinite(convergence).all() or not np.isfinite(best_position).all():
        raise RuntimeError(
            f"nonfinite result {algorithm}/F{function_number}/run{run}"
        )
    reevaluated = float(F(
        np.ascontiguousarray(best_position.reshape(1, -1)),
        matrix,
        shift,
        function_number,
        shuffle,
        BENCHMARK_ID,
    )[0, 0])
    agreement = abs(reevaluated - best_fitness)
    if agreement > 1e-8 * (1.0 + abs(best_fitness)):
        raise RuntimeError(
            f"fitness-position mismatch {algorithm}/F{function_number}/run{run}"
        )

    metadata_path, data_path = result_paths(
        output_root, algorithm, function_number, run
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = data_path.with_suffix(".npz.tmp")
    with temporary_data.open("wb") as handle:
        np.savez_compressed(
            handle,
            best_position=best_position,
            best_fitness=np.asarray([best_fitness]),
        )
    os.replace(temporary_data, data_path)

    metadata = {
        "algorithm": algorithm,
        "benchmark": BENCHMARK,
        "function": int(function_number),
        "run": int(run),
        "dimension": DIMENSION,
        "population_size": POPULATION_SIZE,
        "max_fes": int(max_fes),
        "final_fes": int(evaluation_axis[-1]),
        "seed": int(seed),
        "initial_population_sha256": initial_hash,
        "best_fitness": best_fitness,
        "reevaluated_best_fitness": reevaluated,
        "fitness_position_abs_error": agreement,
        "elapsed_seconds": float(elapsed),
        "result_path": str(data_path.resolve()),
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary_metadata, metadata_path)
    return metadata


def run_group(task: tuple) -> list[dict]:
    output_root, algorithm, function_number, runs, max_fes = task
    rows = []
    for run in runs:
        rows.append(run_one(
            output_root, algorithm, function_number, run, max_fes
        ))
    return rows


def build_tasks(output_root: Path, algorithms: tuple[str, ...],
                functions: tuple[int, ...], runs: int, max_fes: int,
                chunk_size: int) -> tuple[list[tuple], int]:
    tasks = []
    complete = 0
    for algorithm in algorithms:
        for function_number in functions:
            pending = []
            for run in range(runs):
                if metadata_complete(
                    output_root, algorithm, function_number, run, max_fes
                ):
                    complete += 1
                else:
                    pending.append(run)
            for start in range(0, len(pending), chunk_size):
                tasks.append((
                    str(output_root),
                    algorithm,
                    function_number,
                    tuple(pending[start:start + chunk_size]),
                    max_fes,
                ))
    return tasks, complete


def write_protocol(output_root: Path, algorithms: tuple[str, ...],
                   functions: tuple[int, ...], runs: int,
                   max_fes: int) -> None:
    source_files = (
        "run_high_dimensional_scalability.py",
        "IIGO_Final.py",
        "ExperimentAlgorithms.py",
        "Interface.py",
        "experiment_config.py",
        "elements.py",
        "CEC2017SO_BC.py",
        "de_common.py",
        "CSO.py",
        "LSHADE_cnEpSin.py",
        "jSO.py",
    )
    protocol = {
        "experiment_id": "iigo_high_dimensional_scalability_d100",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": BENCHMARK,
        "functions": list(functions),
        "function_note": "Official CEC 2017 set excluding withdrawn F2",
        "dimension": DIMENSION,
        "initial_population_size": POPULATION_SIZE,
        "max_fes": max_fes,
        "max_fes_rule": "10000*D" if max_fes == 10_000 * DIMENSION else "custom",
        "independent_runs": runs,
        "algorithms": list(algorithms),
        "paired_design": (
            "For each function and run, all algorithms use the same seed "
            "and bit-identical initial population."
        ),
        "base_run_seeds": list(RUN_SEEDS[:runs]),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "numba": numba.__version__,
        },
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in source_files if (ROOT / name).exists()
        },
    }
    (output_root / "protocol_manifest.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    seed_rows = []
    for function_number in functions:
        for run in range(runs):
            seed_rows.append({
                "benchmark": BENCHMARK,
                "function": function_number,
                "run": run,
                "base_seed": RUN_SEEDS[run],
                "derived_seed": derive_seed(BENCHMARK, function_number, run),
            })
    with (output_root / "seed_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)


def load_rows(output_root: Path) -> list[dict]:
    rows = []
    for path in output_root.glob("*/F*/*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "algorithm" in row and "best_fitness" in row:
            rows.append(row)
    rows.sort(key=lambda row: (
        row["algorithm"], row["function"], row["run"]
    ))
    return rows


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    total = len(ordered)
    adjusted = {}
    previous = 0.0
    for index, name in enumerate(ordered):
        value = min(1.0, (total - index) * p_values[name])
        previous = max(previous, value)
        adjusted[name] = previous
    return adjusted


def validate_and_summarize(output_root: Path, algorithms: tuple[str, ...],
                           functions: tuple[int, ...], runs: int,
                           max_fes: int) -> dict:
    rows = load_rows(output_root)
    expected = len(algorithms) * len(functions) * runs
    failures = []
    if len(rows) != expected:
        failures.append(f"row_count={len(rows)}, expected={expected}")

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(output_root / "full_results.csv", index=False,
                     encoding="utf-8-sig")
    condition_map = {}
    for row in rows:
        key = (int(row["function"]), int(row["run"]))
        item = condition_map.setdefault(
            key, {"seeds": set(), "hashes": set(), "algorithms": set()}
        )
        item["seeds"].add(int(row["seed"]))
        item["hashes"].add(row["initial_population_sha256"])
        item["algorithms"].add(row["algorithm"])
        if int(row["final_fes"]) != max_fes:
            failures.append(f"budget={row['algorithm']}/F{row['function']}/run{row['run']}")
        if not np.isfinite(float(row["best_fitness"])):
            failures.append(f"nonfinite={row['algorithm']}/F{row['function']}/run{row['run']}")
    for key, item in condition_map.items():
        if (
            len(item["seeds"]) != 1
            or len(item["hashes"]) != 1
            or item["algorithms"] != set(algorithms)
        ):
            failures.append(f"unpaired={key}")

    if failures or frame.empty:
        return {
            "status": "FAIL",
            "expected_runs": expected,
            "completed_runs": len(rows),
            "paired_conditions": len(condition_map),
            "failures": failures,
        }

    summary = frame.groupby(["algorithm", "function"], sort=False)[
        "best_fitness"
    ].agg(["mean", "std", "median", "min", "max", "count"]).reset_index()
    summary.to_csv(output_root / "function_summary.csv", index=False,
                   encoding="utf-8-sig")

    mean_matrix = summary.pivot(
        index="function", columns="algorithm", values="mean"
    ).reindex(index=functions, columns=algorithms)
    rank_matrix = mean_matrix.apply(
        lambda row: pd.Series(
            rankdata(row.to_numpy(dtype=float), method="average"),
            index=row.index,
        ),
        axis=1,
    )
    average_ranks = rank_matrix.mean(axis=0).sort_values()
    ranking = pd.DataFrame({
        "algorithm": average_ranks.index,
        "average_rank": average_ranks.values,
        "overall_rank": np.arange(1, len(average_ranks) + 1),
    })
    ranking.to_csv(output_root / "average_ranking.csv", index=False,
                   encoding="utf-8-sig")

    if len(algorithms) >= 3 and len(functions) >= 2:
        friedman_stat, friedman_p = friedmanchisquare(
            *[mean_matrix[name].to_numpy() for name in algorithms]
        )
    else:
        friedman_stat, friedman_p = float("nan"), float("nan")
    raw_p = {}
    wtl_rows = []
    reference = mean_matrix["IIGO"].to_numpy()
    for algorithm in algorithms:
        if algorithm == "IIGO":
            continue
        competitor = mean_matrix[algorithm].to_numpy()
        tolerance = 1e-12 * np.maximum(
            1.0, np.maximum(np.abs(reference), np.abs(competitor))
        )
        wins = int(np.sum(reference < competitor - tolerance))
        ties = int(np.sum(np.abs(reference - competitor) <= tolerance))
        losses = int(np.sum(reference > competitor + tolerance))
        try:
            _, p_value = wilcoxon(
                reference, competitor, alternative="two-sided",
                zero_method="pratt"
            )
            p_value = float(p_value)
        except ValueError:
            p_value = 1.0
        raw_p[algorithm] = p_value
        wtl_rows.append({
            "comparison": f"IIGO vs {algorithm}",
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "wilcoxon_p": p_value,
        })
    adjusted = holm_adjust(raw_p)
    for row in wtl_rows:
        algorithm = row["comparison"].replace("IIGO vs ", "")
        row["holm_adjusted_p"] = adjusted[algorithm]
    wtl = pd.DataFrame(wtl_rows)
    wtl.to_csv(output_root / "iigo_wtl_wilcoxon_holm.csv", index=False,
               encoding="utf-8-sig")

    with pd.ExcelWriter(
        output_root / "high_dimensional_scalability_results.xlsx",
        engine="openpyxl",
    ) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Raw runs", index=False)
        summary.to_excel(writer, sheet_name="Function summary", index=False)
        mean_matrix.reset_index().to_excel(
            writer, sheet_name="Mean fitness", index=False
        )
        rank_matrix.reset_index().to_excel(
            writer, sheet_name="Function ranks", index=False
        )
        ranking.to_excel(writer, sheet_name="Average ranking", index=False)
        wtl.to_excel(writer, sheet_name="IIGO WTL", index=False)
        pd.DataFrame([{
            "friedman_statistic": float(friedman_stat),
            "friedman_p": float(friedman_p),
        }]).to_excel(writer, sheet_name="Friedman", index=False)

    return {
        "status": "PASS",
        "expected_runs": expected,
        "completed_runs": len(rows),
        "paired_conditions": len(condition_map),
        "friedman_statistic": float(friedman_stat),
        "friedman_p": float(friedman_p),
        "average_ranking": ranking.to_dict(orient="records"),
        "iigo_wtl": wtl_rows,
        "failures": [],
    }


def parse_csv_tuple(text: str, cast) -> tuple:
    return tuple(cast(item.strip()) for item in text.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--max-fes", type=int, default=MAX_FES)
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument(
        "--functions", default=",".join(str(value) for value in FUNCTIONS)
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    algorithms = parse_csv_tuple(args.algorithms, str)
    functions = parse_csv_tuple(args.functions, int)
    if not 1 <= args.runs <= len(RUN_SEEDS):
        raise SystemExit(f"runs must be 1..{len(RUN_SEEDS)}")
    if args.max_fes < POPULATION_SIZE:
        raise SystemExit("max-fes must cover population initialization")
    if any(name not in ALGORITHMS for name in algorithms):
        raise SystemExit("unsupported algorithm selection")
    if any(fn not in FUNCTIONS for fn in functions):
        raise SystemExit("unsupported CEC 2017 function selection")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "protocol_manifest.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        signature = (
            previous.get("dimension"),
            previous.get("initial_population_size"),
            previous.get("max_fes"),
            previous.get("independent_runs"),
            tuple(previous.get("algorithms", [])),
            tuple(previous.get("functions", [])),
        )
        current = (
            DIMENSION, POPULATION_SIZE, args.max_fes, args.runs,
            algorithms, functions,
        )
        if signature != current:
            raise SystemExit(
                "output directory contains a different experiment protocol"
            )
    write_protocol(
        output_root, algorithms, functions, args.runs, args.max_fes
    )

    tasks, complete = build_tasks(
        output_root, algorithms, functions, args.runs,
        args.max_fes, max(1, args.chunk_size),
    )
    pending = sum(len(task[3]) for task in tasks)
    total = len(algorithms) * len(functions) * args.runs
    print(
        f"Scalability experiment: algorithms={len(algorithms)}, "
        f"functions={len(functions)}, runs={args.runs}, total={total}",
        flush=True,
    )
    print(
        f"D={DIMENSION}, NP_init={POPULATION_SIZE}, MaxFEs={args.max_fes}, "
        f"workers={max(1, args.workers)}, pending={pending}, resumed={complete}",
        flush=True,
    )
    started = time.time()
    completed = 0
    log_path = output_root / "run_log.jsonl"
    if tasks:
        context = mp.get_context("spawn")
        with context.Pool(processes=max(1, args.workers)) as pool, log_path.open(
            "a", encoding="utf-8"
        ) as log_handle:
            for batch in pool.imap_unordered(run_group, tasks):
                completed += len(batch)
                for row in batch:
                    log_handle.write(json.dumps({
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "algorithm": row["algorithm"],
                        "function": row["function"],
                        "run": row["run"],
                        "best_fitness": row["best_fitness"],
                        "elapsed_seconds": row["elapsed_seconds"],
                    }, ensure_ascii=False) + "\n")
                log_handle.flush()
                elapsed = time.time() - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (pending - completed) / rate if rate > 0 else 0.0
                if completed % 10 == 0 or completed == pending:
                    print(
                        f"progress={completed}/{pending} "
                        f"total={complete + completed}/{total} "
                        f"elapsed={elapsed / 60:.1f}min ETA={eta / 60:.1f}min",
                        flush=True,
                    )

    report = validate_and_summarize(
        output_root, algorithms, functions, args.runs, args.max_fes
    )
    report["duration_seconds_this_invocation"] = time.time() - started
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output_root / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
