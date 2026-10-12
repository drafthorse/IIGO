"""Run and validate CEC benchmark experiments.

The default command runs a small acceptance test:
all configured algorithms x 3 CEC suites x 1 function x 2 independent runs.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numba
import numpy as np

import Interface
from elements import F
from experiment_config import (
    ALGORITHMS, BENCHMARK_IDS, DIMENSION, MAX_FES, POPULATION_SIZE,
    PAPER_FUNCTIONS, RUN_SEEDS, derive_seed,
)


ACCEPTANCE_FUNCTIONS = {"CEC2014": 30, "CEC2017": 29, "CEC2022": 12}


def check_cec_zero_reference():
    checks = []
    for benchmark, function_number in ACCEPTANCE_FUNCTIONS.items():
        matrix, shift, shuffle = Interface.load_cec_input(
            benchmark, function_number, DIMENSION)
        point = np.ascontiguousarray(shift[:DIMENSION].reshape(1, -1))
        value = float(F(point, matrix, shift, function_number, shuffle,
                        BENCHMARK_IDS[benchmark])[0, 0])
        checks.append({
            "benchmark": benchmark,
            "function": function_number,
            "first_shift_error": value,
            "near_zero": bool(abs(value) <= 1e-7),
        })
    return checks


def build_tasks(output_root):
    tasks = []
    for optimizer in ALGORITHMS:
        for benchmark, function_number in ACCEPTANCE_FUNCTIONS.items():
            tasks.append((
                DIMENSION, POPULATION_SIZE, function_number, optimizer,
                benchmark, MAX_FES, str(output_root), (0, 1),
            ))
    return tasks


def _result_files(output_root, optimizer, benchmark, function_number, run_number):
    paper_number = PAPER_FUNCTIONS[benchmark][function_number]
    directory = output_root / optimizer / benchmark / f"F{paper_number}"
    stem = f"{optimizer}_F{paper_number}_{run_number}"
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def _is_complete(output_root, optimizer, benchmark, function_number, run_number):
    result_path, metadata_path = _result_files(
        output_root, optimizer, benchmark, function_number, run_number)
    if not result_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("optimization_fes") != MAX_FES
                or metadata.get("dimension") != DIMENSION
                or metadata.get("population_size") != POPULATION_SIZE
                or metadata.get("seed") != derive_seed(
                    benchmark, function_number, run_number)):
            return False
        with np.load(result_path) as data:
            return (data["fes"].size > 0
                    and int(data["fes"][-1]) == MAX_FES
                    and data["best_position"].shape == (DIMENSION,))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def build_full_tasks(output_root, chunk_size=5):
    tasks = []
    skipped = 0
    for optimizer in ALGORITHMS:
        for benchmark, functions in PAPER_FUNCTIONS.items():
            for function_number in functions:
                pending = []
                for run_number in range(30):
                    if _is_complete(
                            output_root, optimizer, benchmark,
                            function_number, run_number):
                        skipped += 1
                    else:
                        pending.append(run_number)
                for start in range(0, len(pending), chunk_size):
                    tasks.append((
                        DIMENSION, POPULATION_SIZE, function_number, optimizer,
                        benchmark, MAX_FES, str(output_root),
                        tuple(pending[start:start+chunk_size]),
                    ))
    return tasks, skipped


def load_metadata_rows(output_root):
    rows = []
    for path in output_root.rglob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "optimizer" in row and "result_path" in row:
            rows.append(row)
    rows.sort(key=lambda x: (
        x["optimizer"], x["paper_function"], x["run_number"]))
    return rows


def validate_full_results(metadata_rows):
    failures = []
    keys = set()
    pairing = {}
    expected = len(ALGORITHMS)*sum(len(x) for x in PAPER_FUNCTIONS.values())*30
    if len(metadata_rows) != expected:
        failures.append(f"metadata row count {len(metadata_rows)} != {expected}")
    for row in metadata_rows:
        key = (row["optimizer"], row["benchmark"],
               row["cec_function"], row["run_number"])
        if key in keys:
            failures.append(f"duplicate result {key}")
        keys.add(key)
        result_path = Path(row["result_path"])
        try:
            with np.load(result_path) as data:
                curve = data["convergence"]
                fes = data["fes"]
                best_position = data["best_position"]
                best_fitness = float(data["best_fitness"][0])
        except (OSError, ValueError, KeyError) as error:
            failures.append(f"{key}: unreadable result: {error}")
            continue
        if fes.size == 0 or int(fes[-1]) != MAX_FES:
            failures.append(f"{key}: final FE is not {MAX_FES}")
        if int(row.get("population_size", -1)) != POPULATION_SIZE:
            failures.append(
                f"{key}: population is {row.get('population_size')}, "
                f"expected {POPULATION_SIZE}"
            )
        if np.any(np.diff(fes) <= 0):
            failures.append(f"{key}: FE trajectory is not strictly increasing")
        if curve.size != fes.size:
            failures.append(f"{key}: curve and FE lengths differ")
        if curve.size > 1:
            tolerance = 1e-10*np.maximum(1.0, np.abs(curve[:-1]))
            if np.any(np.diff(curve) > tolerance):
                failures.append(f"{key}: best-so-far curve increases")
        if (not np.all(np.isfinite(curve))
                or not np.all(np.isfinite(best_position))):
            failures.append(f"{key}: non-finite output")
        if not np.isclose(curve[-1], best_fitness, rtol=1e-12, atol=1e-12):
            failures.append(f"{key}: curve and best fitness differ")
        tolerance = 1e-8*(1.0+abs(best_fitness))
        if row["fitness_position_abs_error"] > tolerance:
            failures.append(f"{key}: best individual/fitness mismatch")
        pair_key = (row["benchmark"], row["cec_function"], row["run_number"])
        item = pairing.setdefault(
            pair_key, {"seeds": set(), "hashes": set(), "algorithms": set()})
        item["seeds"].add(row["seed"])
        item["hashes"].add(row["initial_population_sha256"])
        item["algorithms"].add(row["optimizer"])
    for key, item in pairing.items():
        if (len(item["seeds"]) != 1 or len(item["hashes"]) != 1
                or item["algorithms"] != set(ALGORITHMS)):
            failures.append(f"{key}: paired initialization is incomplete or inconsistent")
    if len(pairing) != 30*30:
        failures.append(f"paired condition count {len(pairing)} != 900")
    return failures, len(pairing)


def write_full_tables(metadata_rows, output_root):
    index_path = output_root / "full_results.csv"
    with index_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    grouped = {}
    for row in metadata_rows:
        key = (row["optimizer"], row["benchmark"],
               row["cec_function"], row["paper_function"])
        grouped.setdefault(key, []).append(float(row["best_fitness"]))
    summary_rows = []
    for key, values in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][3])):
        values = np.asarray(values)
        summary_rows.append({
            "optimizer": key[0], "benchmark": key[1],
            "cec_function": key[2], "paper_function": key[3],
            "runs": values.size, "mean": float(np.mean(values)),
            "std": float(np.std(values)), "median": float(np.median(values)),
            "min": float(np.min(values)), "max": float(np.max(values)),
        })
    summary_path = output_root / "full_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return index_path, summary_path, grouped


def write_convergence_summaries(metadata_rows):
    grouped = {}
    for row in metadata_rows:
        key = (row["optimizer"], row["benchmark"], row["paper_function"])
        grouped.setdefault(key, []).append(row)
    checkpoints = np.concatenate((
        np.asarray([POPULATION_SIZE], dtype=np.int64),
        np.arange(1000, MAX_FES+1, 1000, dtype=np.int64),
    ))
    written = 0
    for rows in grouped.values():
        sampled = np.empty((len(rows), checkpoints.size))
        for i, row in enumerate(sorted(rows, key=lambda x: x["run_number"])):
            with np.load(row["result_path"]) as data:
                fes = data["fes"]
                curve = data["convergence"]
            indices = np.searchsorted(fes, checkpoints, side="right")-1
            indices = np.maximum(indices, 0)
            sampled[i] = curve[indices]
        directory = Path(rows[0]["result_path"]).parent
        np.savez_compressed(
            directory / "convergence_summary.npz",
            fes=checkpoints, mean=np.mean(sampled, axis=0),
            std=np.std(sampled, axis=0), median=np.median(sampled, axis=0),
            minimum=np.min(sampled, axis=0), maximum=np.max(sampled, axis=0),
        )
        written += 1
    return written


def validate_results(metadata_rows, output_root, zero_checks):
    failures = []
    grouped = {}
    for row in metadata_rows:
        result_path = Path(row["result_path"])
        if not result_path.exists():
            failures.append(f"missing result: {result_path}")
            continue
        with np.load(result_path) as data:
            curve = data["convergence"]
            fes = data["fes"]
            best_position = data["best_position"]
            best_fitness = float(data["best_fitness"][0])
        label = (row["optimizer"], row["benchmark"], row["cec_function"], row["run_number"])
        if fes.size == 0 or int(fes[-1]) != MAX_FES:
            failures.append(f"{label}: final FE is {fes[-1] if fes.size else 'empty'}")
        if np.any(np.diff(fes) <= 0):
            failures.append(f"{label}: FE trajectory is not strictly increasing")
        tolerance = 1e-10*np.maximum(1.0, np.abs(curve[:-1]))
        if curve.size > 1 and np.any(np.diff(curve) > tolerance):
            failures.append(f"{label}: best-so-far curve increases")
        if not np.all(np.isfinite(curve)) or not np.all(np.isfinite(best_position)):
            failures.append(f"{label}: non-finite output")
        if not np.isclose(curve[-1], best_fitness, rtol=1e-12, atol=1e-12):
            failures.append(f"{label}: curve and saved best fitness differ")
        audit_tolerance = 1e-8*(1.0+abs(best_fitness))
        if row["fitness_position_abs_error"] > audit_tolerance:
            failures.append(f"{label}: best position re-evaluation mismatch")
        key = (row["benchmark"], row["cec_function"], row["run_number"])
        grouped.setdefault(key, {"seeds": set(), "hashes": set()})
        grouped[key]["seeds"].add(row["seed"])
        grouped[key]["hashes"].add(row["initial_population_sha256"])

    pairing = []
    for key, values in grouped.items():
        paired = len(values["seeds"]) == 1 and len(values["hashes"]) == 1
        pairing.append({
            "benchmark": key[0], "function": key[1], "run_number": key[2],
            "seed": next(iter(values["seeds"])),
            "initial_population_sha256": next(iter(values["hashes"])),
            "paired_across_algorithms": paired,
        })
        if not paired:
            failures.append(f"{key}: algorithms did not share seed/initial population")
    for check in zero_checks:
        if not check["near_zero"]:
            failures.append(
                f"{check['benchmark']} F{check['function']}: first shift error is "
                f"{check['first_shift_error']}")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "protocol": {
            "dimension": DIMENSION,
            "population_size": POPULATION_SIZE,
            "max_fes": MAX_FES,
            "algorithms": list(ALGORITHMS),
            "acceptance_functions": ACCEPTANCE_FUNCTIONS,
            "runs": [0, 1],
            "task_count": len(metadata_rows),
            "base_run_seeds": list(RUN_SEEDS),
            "note": "diagnostic_fes=1 re-evaluates the returned best point after the 300000-FE optimization budget",
        },
        "cec_zero_reference_checks": zero_checks,
        "paired_initialization_checks": pairing,
        "failures": failures,
    }
    report_path = output_root / "acceptance_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output_root / "acceptance_results.csv"
    if metadata_rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metadata_rows)
    return report, report_path, csv_path


def run_acceptance(workers):
    output_root = Path("data4") / "acceptance" / "30d_300000fes"
    output_root.mkdir(parents=True, exist_ok=True)
    seed_manifest = {
        benchmark: {
            str(run): derive_seed(benchmark, function_number, run)
            for run in range(30)
        }
        for benchmark, function_number in ACCEPTANCE_FUNCTIONS.items()
    }
    (output_root / "seed_manifest.json").write_text(
        json.dumps(seed_manifest, indent=2), encoding="utf-8")

    zero_checks = check_cec_zero_reference()
    tasks = build_tasks(output_root)
    print(f"Acceptance matrix: {len(tasks)} grouped tasks, {len(tasks)*2} runs")
    print(f"D={DIMENSION}, NP={POPULATION_SIZE}, MaxFEs={MAX_FES}, workers={workers}")
    started = time.time()
    rows = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        for completed, pair in enumerate(pool.imap_unordered(Interface.run_pair, tasks), start=1):
            rows.extend(pair)
            elapsed = time.time()-started
            print(f"completed {completed}/{len(tasks)} groups ({2*completed} runs), {elapsed/60:.1f} min")
    rows.sort(key=lambda x: (x["optimizer"], x["benchmark"], x["run_number"]))
    report, report_path, csv_path = validate_results(rows, output_root, zero_checks)
    print(f"Acceptance status: {report['status']}")
    print(f"Report: {report_path.resolve()}")
    print(f"Results: {csv_path.resolve()}")
    if report["failures"]:
        for failure in report["failures"]:
            print(f"FAIL: {failure}")
        raise SystemExit(1)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_full(workers, chunk_size, output_root=None, population_size=None):
    global POPULATION_SIZE
    if population_size is not None:
        POPULATION_SIZE = max(4, int(population_size))
    output_root = (Path(output_root) if output_root is not None else
                   Path("data4") / "full" / "30d_300000fes")
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    source_files = (
        "Platform.py", "Interface.py", "ExperimentAlgorithms.py",
        "IIGO_Final.py",
        "experiment_config.py", "elements.py", "CEC2014SO_BC.py",
        "CEC2017SO_BC.py", "CEC2022SO_BC.py",
        "de_common.py", "CSO.py", "LSHADE_cnEpSin.py", "jSO.py",
    )
    manifest = {
        "experiment_id": "cec_full_30d_300000fes",
        "started_at": started_at.isoformat(),
        "command": (
            f"python Platform.py --mode full --workers {workers} "
            f"--chunk-size {chunk_size} --population-size {POPULATION_SIZE} "
            f"--output-root \"{output_root}\""
        ),
        "dimension": DIMENSION,
        "population_size": POPULATION_SIZE,
        "population_protocol": (
            f"All algorithms start from the same NP_init={POPULATION_SIZE}; "
            "population-reduction algorithms retain their implemented reduction to 4."
        ),
        "max_fes": MAX_FES,
        "run_count": 30,
        "algorithm_count": len(ALGORITHMS),
        "algorithms": list(ALGORITHMS),
        "paper_functions": PAPER_FUNCTIONS,
        "expected_runs": len(ALGORITHMS)*30*30,
        "base_run_seeds": list(RUN_SEEDS),
        "python": sys.version,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "source_sha256": {path: _sha256(path) for path in source_files},
        "budget_note": "300000 optimization evaluations plus one separate post-run diagnostic re-evaluation",
        "lshade_cnepsin_port_note": (
            "Python port of the official MATLAB mechanisms, adapted to the "
            f"shared NP={POPULATION_SIZE} initialization and exact MaxFEs protocol; the "
            "official standalone MATLAB default NP=18*D is intentionally not used."
        ),
    }
    manifest_path = output_root / "protocol_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("algorithms") != list(ALGORITHMS):
            old_count = previous.get("algorithm_count", "previous")
            backup = output_root / f"protocol_manifest_{old_count}alg.json"
            if not backup.exists():
                shutil.copy2(manifest_path, backup)
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    seed_manifest = {
        benchmark: {
            str(function_number): {
                str(run): derive_seed(benchmark, function_number, run)
                for run in range(30)
            }
            for function_number in functions
        }
        for benchmark, functions in PAPER_FUNCTIONS.items()
    }
    (output_root / "seed_manifest.json").write_text(
        json.dumps(seed_manifest, indent=2), encoding="utf-8")

    zero_checks = check_cec_zero_reference()
    tasks, skipped = build_full_tasks(output_root, chunk_size)
    pending_runs = sum(len(task[-1]) for task in tasks)
    expected_runs = len(ALGORITHMS)*30*30
    print(
        f"Full matrix: {len(ALGORITHMS)} algorithms x 30 functions x 30 runs "
        f"= {expected_runs} runs")
    print(
        f"D={DIMENSION}, NP={POPULATION_SIZE}, MaxFEs={MAX_FES}, "
        f"workers={workers}, pending={pending_runs}, resumed={skipped}")
    started = time.time()
    log_path = output_root / "run_log.jsonl"
    if tasks:
        context = multiprocessing.get_context("spawn")
        completed_runs = 0
        with context.Pool(processes=workers) as pool, log_path.open(
                "a", encoding="utf-8") as log_handle:
            for completed_groups, batch in enumerate(
                    pool.imap_unordered(Interface.run_pair, tasks), start=1):
                completed_runs += len(batch)
                for row in batch:
                    log_handle.write(json.dumps({
                        "completed_at": datetime.now().astimezone().isoformat(),
                        "optimizer": row["optimizer"],
                        "benchmark": row["benchmark"],
                        "cec_function": row["cec_function"],
                        "paper_function": row["paper_function"],
                        "run_number": row["run_number"],
                        "seed": row["seed"],
                        "optimization_fes": row["optimization_fes"],
                        "best_fitness": row["best_fitness"],
                        "result_path": row["result_path"],
                    })+"\n")
                log_handle.flush()
                if completed_groups % 10 == 0 or completed_groups == len(tasks):
                    elapsed = time.time()-started
                    rate = completed_runs/elapsed if elapsed > 0 else 0.0
                    remaining = (pending_runs-completed_runs)/rate if rate > 0 else 0.0
                    print(
                        f"completed {completed_runs}/{pending_runs} pending runs "
                        f"({skipped+completed_runs}/{expected_runs} total), "
                        f"elapsed={elapsed/60:.1f} min, ETA={remaining/60:.1f} min")

    print("Execution complete; loading metadata and validating all outputs...")
    metadata_rows = load_metadata_rows(output_root)
    failures, paired_conditions = validate_full_results(metadata_rows)
    for check in zero_checks:
        if not check["near_zero"]:
            failures.append(
                f"{check['benchmark']} F{check['function']} zero-reference check failed")

    index_path = None
    summary_path = None
    convergence_files = 0
    if metadata_rows:
        index_path, summary_path, _ = write_full_tables(metadata_rows, output_root)
    if not failures:
        print("Validation passed; generating convergence summaries for plotting...")
        convergence_files = write_convergence_summaries(metadata_rows)
    elapsed = time.time()-started
    report = {
        "status": "PASS" if not failures else "FAIL",
        "completed_at": datetime.now().astimezone().isoformat(),
        "duration_seconds_this_invocation": elapsed,
        "expected_runs": expected_runs,
        "metadata_rows": len(metadata_rows),
        "paired_function_run_conditions": paired_conditions,
        "convergence_summary_files": convergence_files,
        "cec_zero_reference_checks": zero_checks,
        "failures": failures,
        "output_files": {
            "results_index": str(index_path.resolve()) if index_path else None,
            "statistical_summary": str(summary_path.resolve()) if summary_path else None,
            "run_log": str(log_path.resolve()),
            "protocol_manifest": str((output_root / "protocol_manifest.json").resolve()),
            "seed_manifest": str((output_root / "seed_manifest.json").resolve()),
        },
    }
    report_path = output_root / "full_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_markdown = f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: {datetime.now().date().isoformat()}
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

## Full CEC Experiment Result

- Status: {report['status']}
- Dimension: {DIMENSION}
- Population size: {POPULATION_SIZE}
- MaxFEs: {MAX_FES}
- Algorithms: {len(ALGORITHMS)}
- Paper functions: 30
- Independent runs per condition: 30
- Completed runs: {len(metadata_rows)}/{expected_runs}
- Paired initialization conditions: {paired_conditions}/900
- Convergence summary files: {convergence_files}/{len(ALGORITHMS)*30}
- Duration of final invocation: {elapsed/60:.2f} minutes
- Failures: {len(failures)}

Raw `.npz` files contain each run's convergence curve, FE trajectory, best
position and best fitness. `full_results.csv` is the per-run index,
`full_summary.csv` contains the 30-run descriptive statistics, and each
algorithm/function folder contains `convergence_summary.npz` at common FE
checkpoints for direct plotting.
"""
    (output_root / "FULL_EXPERIMENT_SUMMARY.md").write_text(
        summary_markdown, encoding="utf-8")
    print(f"Full experiment status: {report['status']}")
    print(f"Report: {report_path.resolve()}")
    if failures:
        for failure in failures[:50]:
            print(f"FAIL: {failure}")
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--mode", choices=("acceptance", "full"),
                        default="acceptance")
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Optional isolated output directory for a full run.",
    )
    parser.add_argument(
        "--population-size", type=int, default=None,
        help="Override the shared initial population size for a full run.",
    )
    args = parser.parse_args()
    if args.mode == "full":
        run_full(
            max(1, args.workers), max(1, args.chunk_size), args.output_root,
            args.population_size,
        )
    else:
        run_acceptance(max(1, args.workers))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
