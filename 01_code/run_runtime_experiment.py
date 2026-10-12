"""Sequential, post-warm-up execution-time experiment for final IIGO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np

import Interface
from ExperimentAlgorithms import get_f
from experiment_config import ALGORITHMS, BENCHMARK_IDS
from IIGO_Ablations import ABLATION_FUNCTIONS


ROOT = Path(__file__).resolve().parent
DIMENSION = 30
POPULATION_SIZE = 10
MAX_FES = 300_000
CASES = (
    ("CEC2014", 30, 11),
    ("CEC2017", 29, 22),
    ("CEC2022", 12, 30),
)
TIMED_ALGORITHMS = tuple(ALGORITHMS) + (
    "IIGO_NoWindow", "IIGO_NoIGE", "IIGO_NoStage",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def function_for(name: str):
    if name in ABLATION_FUNCTIONS:
        return ABLATION_FUNCTIONS[name]
    return get_f(name)


def run_call(name: str, benchmark: str, function_number: int, run: int,
             max_fes: int):
    matrix, shift, shuffle = Interface.load_cec_input(
        benchmark, function_number, DIMENSION
    )
    seed, initial_population, initial_hash = Interface.make_initial_population(
        benchmark, function_number, run, POPULATION_SIZE, DIMENSION
    )
    function = function_for(name)
    started = time.perf_counter()
    curve, fes, best_position, best_fitness = function(
        max_fes,
        POPULATION_SIZE,
        -100.0,
        100.0,
        DIMENSION,
        matrix,
        shift,
        function_number,
        shuffle,
        BENCHMARK_IDS[benchmark],
        seed,
        initial_population,
    )
    elapsed = time.perf_counter() - started
    return {
        "algorithm": name,
        "benchmark": benchmark,
        "cec_function": function_number,
        "run": run,
        "seed": seed,
        "initial_population_sha256": initial_hash,
        "max_fes": max_fes,
        "final_fes": int(fes[-1]),
        "elapsed_seconds": float(elapsed),
        "best_fitness": float(best_fitness),
        "curve_points": int(len(curve)),
        "position_finite": bool(np.isfinite(best_position).all()),
    }


def write_manifest(output_root: Path, repeats: int, warmup_fes: int,
                   max_fes: int) -> None:
    source_files = (
        "IIGO_Final.py", "IIGO_Ablations.py", "ExperimentAlgorithms.py",
        "Interface.py", "experiment_config.py", "elements.py",
        "run_runtime_experiment.py",
    )
    manifest = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run+validate",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "verification_status": "PENDING",
            "version_label": "final_iigo_runtime_v1",
        },
        "timing_scope": (
            "optimizer call only; excludes input loading, initial-population "
            "generation, result serialization and one-time JIT warm-up"
        ),
        "execution_policy": (
            "single process, sequential calls, fixed randomized order within "
            "each case/repeat block; run only when no benchmark worker pool is active"
        ),
        "algorithms": TIMED_ALGORITHMS,
        "cases": CASES,
        "repeats": repeats,
        "warmup_fes": warmup_fes,
        "timed_fes": max_fes,
        "dimension": DIMENSION,
        "population_size": POPULATION_SIZE,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "numba": numba.__version__,
        },
        "source_sha256": {
            name: sha256(ROOT / name) for name in source_files
        },
    }
    (output_root / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup-fes", type=int, default=5_000)
    parser.add_argument("--max-fes", type=int, default=MAX_FES)
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "data4" / "final_iigo_runtime" / "30d_300000fes",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup_fes < POPULATION_SIZE:
        raise SystemExit("invalid repeats or warmup-fes")
    if args.max_fes < POPULATION_SIZE:
        raise SystemExit("max-fes must cover initialization")

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_manifest(
        args.output_root, args.repeats, args.warmup_fes, args.max_fes
    )
    print(
        f"Runtime experiment: algorithms={len(TIMED_ALGORITHMS)}, "
        f"cases={len(CASES)}, repeats={args.repeats}, "
        f"timed_calls={len(TIMED_ALGORITHMS)*len(CASES)*args.repeats}",
        flush=True,
    )

    # Compile and initialize each executable path before any timing begins.
    for index, name in enumerate(TIMED_ALGORITHMS, start=1):
        row = run_call(name, CASES[0][0], CASES[0][1], 0, args.warmup_fes)
        if row["final_fes"] != args.warmup_fes or not row["position_finite"]:
            raise RuntimeError(f"warm-up validation failed for {name}")
        print(f"warmup={index}/{len(TIMED_ALGORITHMS)} algorithm={name}", flush=True)

    rows = []
    order_rng = np.random.default_rng(20260828)
    total = len(TIMED_ALGORITHMS) * len(CASES) * args.repeats
    started_all = time.time()
    for repeat in range(args.repeats):
        for benchmark, function_number, paper_function in CASES:
            order = list(TIMED_ALGORITHMS)
            order_rng.shuffle(order)
            for name in order:
                row = run_call(
                    name, benchmark, function_number, repeat, args.max_fes
                )
                row["paper_function"] = paper_function
                row["timing_repeat"] = repeat
                rows.append(row)
                if len(rows) % 10 == 0 or len(rows) == total:
                    print(
                        f"progress={len(rows)}/{total} last={name}/{benchmark} "
                        f"elapsed={time.time()-started_all:.1f}s",
                        flush=True,
                    )

    failures = []
    for row in rows:
        if row["final_fes"] != args.max_fes:
            failures.append(
                f"budget:{row['algorithm']}/{row['benchmark']}/run{row['run']}"
            )
        if not row["position_finite"] or not np.isfinite(row["best_fitness"]):
            failures.append(
                f"nonfinite:{row['algorithm']}/{row['benchmark']}/run{row['run']}"
            )
        if row["elapsed_seconds"] <= 0.0:
            failures.append(
                f"time:{row['algorithm']}/{row['benchmark']}/run{row['run']}"
            )

    with (args.output_root / "runtime_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault((row["algorithm"], row["benchmark"]), []).append(
            row["elapsed_seconds"]
        )
    summary = []
    for (algorithm, benchmark), values in sorted(grouped.items()):
        values = np.asarray(values, dtype=np.float64)
        summary.append({
            "algorithm": algorithm,
            "benchmark": benchmark,
            "runs": int(values.size),
            "mean_seconds": float(np.mean(values)),
            "std_seconds": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "median_seconds": float(np.median(values)),
            "min_seconds": float(np.min(values)),
            "max_seconds": float(np.max(values)),
        })
    with (args.output_root / "runtime_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    report = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run+validate",
            "verification_status": "VERIFIED" if not failures else "FAILED",
            "version_label": "final_iigo_runtime_v1",
        },
        "status": "PASS" if not failures else "FAIL",
        "timed_calls": len(rows),
        "expected_calls": total,
        "duration_seconds": time.time() - started_all,
        "failures": failures,
    }
    (args.output_root / "runtime_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest_path = args.output_root / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["material_passport"]["verification_status"] = report[
        "material_passport"
    ]["verification_status"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
