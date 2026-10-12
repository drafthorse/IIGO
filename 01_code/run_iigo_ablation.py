"""Run and validate IIGO mechanism-ablation experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import Interface
from elements import F
from experiment_config import BENCHMARK_IDS, PAPER_FUNCTIONS, derive_seed
from IIGO_Ablations import ABLATION_FUNCTIONS


ROOT = Path(__file__).resolve().parent
DIMENSION = 30
POPULATION_SIZE = 10
MAX_FES = 300_000
ALGORITHMS = tuple(ABLATION_FUNCTIONS)
SMOKE_CASES = {"CEC2014": (30,), "CEC2017": (29,), "CEC2022": (12,)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths(output_root: Path, algorithm: str, benchmark: str,
          function_number: int, run: int) -> tuple[Path, Path]:
    paper_function = PAPER_FUNCTIONS[benchmark][function_number]
    folder = output_root / algorithm / benchmark / f"F{paper_function}"
    stem = f"{algorithm}_F{paper_function}_{run}"
    return folder / f"{stem}.npz", folder / f"{stem}.json"


def complete(output_root: Path, algorithm: str, benchmark: str,
             function_number: int, run: int, max_fes: int) -> bool:
    result_path, metadata_path = paths(
        output_root, algorithm, benchmark, function_number, run
    )
    if not result_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(result_path) as payload:
            fes = np.asarray(payload["fes"])
            curve = np.asarray(payload["convergence"])
        return (
            metadata.get("algorithm") == algorithm
            and int(metadata.get("seed", -1))
            == derive_seed(benchmark, function_number, run)
            and int(metadata.get("optimization_fes", -1)) == max_fes
            and int(metadata.get("population_size", -1)) == POPULATION_SIZE
            and fes.size > 0
            and int(fes[-1]) == max_fes
            and fes.shape == curve.shape
        )
    except Exception:
        return False


def run_one(output_root: str, algorithm: str, benchmark: str,
            function_number: int, run: int, max_fes: int) -> dict:
    matrix, shift, shuffle = Interface.load_cec_input(
        benchmark, function_number, DIMENSION
    )
    seed, initial_population, initial_hash = Interface.make_initial_population(
        benchmark, function_number, run, POPULATION_SIZE, DIMENSION
    )
    function = ABLATION_FUNCTIONS[algorithm]
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
    reevaluated = float(
        F(
            np.ascontiguousarray(best_position.reshape(1, -1)),
            matrix,
            shift,
            function_number,
            shuffle,
            BENCHMARK_IDS[benchmark],
        )[0, 0]
    )
    paper_function = PAPER_FUNCTIONS[benchmark][function_number]
    result_path, metadata_path = paths(
        Path(output_root), algorithm, benchmark, function_number, run
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        result_path,
        convergence=np.asarray(curve),
        fes=np.asarray(fes),
        best_position=np.asarray(best_position),
        best_fitness=np.asarray([best_fitness]),
    )
    metadata = {
        "algorithm": algorithm,
        "benchmark": benchmark,
        "benchmark_id": BENCHMARK_IDS[benchmark],
        "cec_function": int(function_number),
        "paper_function": int(paper_function),
        "run": int(run),
        "dimension": DIMENSION,
        "population_size": POPULATION_SIZE,
        "seed": int(seed),
        "initial_population_sha256": initial_hash,
        "optimization_fes": int(fes[-1]),
        "diagnostic_fes": 1,
        "best_fitness": float(best_fitness),
        "reevaluated_best_fitness": reevaluated,
        "fitness_position_abs_error": float(abs(reevaluated - best_fitness)),
        "result_path": str(result_path.resolve()),
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, metadata_path)
    return metadata


def run_chunk(task: tuple) -> list[dict]:
    output_root, algorithm, benchmark, function_number, runs, max_fes = task
    return [
        run_one(
            output_root, algorithm, benchmark, function_number, run, max_fes
        )
        for run in runs
    ]


def build_tasks(output_root: Path, mappings: dict, runs: int, max_fes: int,
                chunk_size: int) -> tuple[list[tuple], int]:
    tasks = []
    already_complete = 0
    for algorithm in ALGORITHMS:
        for benchmark, mapping in mappings.items():
            functions = tuple(mapping) if isinstance(mapping, dict) else mapping
            for function_number in functions:
                pending = []
                for run in range(runs):
                    if complete(
                        output_root, algorithm, benchmark,
                        function_number, run, max_fes
                    ):
                        already_complete += 1
                    else:
                        pending.append(run)
                for start in range(0, len(pending), chunk_size):
                    tasks.append((
                        str(output_root), algorithm, benchmark, function_number,
                        tuple(pending[start:start + chunk_size]), max_fes,
                    ))
    return tasks, already_complete


def write_manifest(output_root: Path, mappings: dict, runs: int,
                   max_fes: int, mode: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    source_files = (
        "IIGO_Final.py", "IIGO_Ablations.py", "run_iigo_ablation.py",
        "Interface.py", "experiment_config.py", "elements.py",
    )
    manifest = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run+validate",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "verification_status": "PENDING",
            "version_label": f"iigo_ablation_{mode}_v1",
        },
        "algorithms": ALGORITHMS,
        "definitions": {
            "IIGO_Full": "unchanged frozen final implementation",
            "IIGO_NoWindow": (
                "FIFO multi-generation trend replaced by signed one-generation "
                "diversity change"
            ),
            "IIGO_NoIGE": (
                "IGE branch replaced by evaluated unchanged target; controller "
                "and stage branch retained"
            ),
            "IIGO_NoStage": (
                "stage-dependent trial replaced by progress-neutral uniform "
                "exploration; controller and IGE retained"
            ),
        },
        "dimension": DIMENSION,
        "population_size": POPULATION_SIZE,
        "max_fes_actual_objective_calls": max_fes,
        "independent_runs": runs,
        "function_mapping": mappings,
        "initialization_policy": (
            "All four variants receive the same seed and byte-identical initial "
            "population within each suite/function/run block."
        ),
        "source_sha256": {
            name: sha256(ROOT / name) for name in source_files
        },
    }
    (output_root / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def validate_and_export(output_root: Path, mappings: dict, runs: int,
                        max_fes: int, mode: str) -> dict:
    failures = []
    rows = []
    expected = len(ALGORITHMS) * sum(len(x) for x in mappings.values()) * runs
    paired = {}
    for algorithm in ALGORITHMS:
        for benchmark, mapping in mappings.items():
            functions = tuple(mapping) if isinstance(mapping, dict) else mapping
            for function_number in functions:
                for run in range(runs):
                    key = f"{algorithm}/{benchmark}/F{function_number}/run{run}"
                    result_path, metadata_path = paths(
                        output_root, algorithm, benchmark, function_number, run
                    )
                    if not metadata_path.exists() or not result_path.exists():
                        failures.append(f"missing:{key}")
                        continue
                    metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    with np.load(result_path) as payload:
                        curve = np.asarray(payload["convergence"])
                        fes = np.asarray(payload["fes"])
                        position = np.asarray(payload["best_position"])
                        stored_best = float(payload["best_fitness"][0])
                    if fes.size == 0 or int(fes[-1]) != max_fes:
                        failures.append(f"budget:{key}")
                    if fes.size > 1 and np.any(np.diff(fes) <= 0):
                        failures.append(f"fes_order:{key}")
                    tolerance = 1e-10 * np.maximum(1.0, np.abs(curve[:-1]))
                    if curve.size > 1 and np.any(np.diff(curve) > tolerance):
                        failures.append(f"curve:{key}")
                    if not np.isfinite(position).all() or not np.isfinite(curve).all():
                        failures.append(f"nonfinite:{key}")
                    best = float(metadata["best_fitness"])
                    if stored_best != best:
                        failures.append(f"stored_best:{key}")
                    if float(metadata["fitness_position_abs_error"]) > 1e-8*(1+abs(best)):
                        failures.append(f"position_fitness:{key}")
                    pair_key = (benchmark, function_number, run)
                    item = paired.setdefault(
                        pair_key, {"seeds": set(), "hashes": set(), "algorithms": set()}
                    )
                    item["seeds"].add(int(metadata["seed"]))
                    item["hashes"].add(metadata["initial_population_sha256"])
                    item["algorithms"].add(algorithm)
                    rows.append(metadata)
    for key, item in paired.items():
        if (
            len(item["seeds"]) != 1
            or len(item["hashes"]) != 1
            or item["algorithms"] != set(ALGORITHMS)
        ):
            failures.append(f"pairing:{key}")

    rows.sort(key=lambda x: (x["algorithm"], x["paper_function"], x["run"]))
    if rows:
        with (output_root / "full_results.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        grouped = {}
        for row in rows:
            group_key = (
                row["algorithm"], row["benchmark"],
                row["cec_function"], row["paper_function"],
            )
            grouped.setdefault(group_key, []).append(row["best_fitness"])
        summary = []
        for group_key, values in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][3])):
            values = np.asarray(values, dtype=np.float64)
            summary.append({
                "algorithm": group_key[0],
                "benchmark": group_key[1],
                "cec_function": group_key[2],
                "paper_function": group_key[3],
                "runs": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
        with (output_root / "full_summary.csv").open(
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
            "version_label": f"iigo_ablation_{mode}_v1",
        },
        "status": "PASS" if not failures else "FAIL",
        "expected_results": expected,
        "validated_results": len(rows),
        "paired_conditions": len(paired),
        "failures": failures[:500],
    }
    (output_root / "full_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest_path = output_root / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["material_passport"]["verification_status"] = report[
        "material_passport"
    ]["verification_status"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--max-fes", type=int, default=MAX_FES)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "data4" / "final_iigo_ablation" / "30d_300000fes",
    )
    args = parser.parse_args()
    mappings = SMOKE_CASES if args.smoke else PAPER_FUNCTIONS
    mode = "smoke" if args.smoke else "full"
    runs = 2 if args.smoke and args.runs == 30 else args.runs
    max_fes = 5_000 if args.smoke and args.max_fes == MAX_FES else args.max_fes
    if args.workers < 1 or args.chunk_size < 1 or runs < 1:
        raise SystemExit("workers, chunk-size and runs must be positive")
    if max_fes < POPULATION_SIZE:
        raise SystemExit("max-fes must cover initialization")
    write_manifest(args.output_root, mappings, runs, max_fes, mode)
    tasks, already_complete = build_tasks(
        args.output_root, mappings, runs, max_fes, args.chunk_size
    )
    total = len(ALGORITHMS) * sum(len(x) for x in mappings.values()) * runs
    print(
        f"IIGO ablation ({mode}): algorithms={ALGORITHMS}, target={total}, "
        f"already_complete={already_complete}, chunks={len(tasks)}",
        flush=True,
    )
    started = time.time()
    completed = already_complete
    if tasks:
        context = mp.get_context("spawn")
        with context.Pool(processes=args.workers) as pool:
            for chunk_index, returned in enumerate(
                pool.imap_unordered(run_chunk, tasks), start=1
            ):
                completed += len(returned)
                print(
                    f"progress={completed}/{total} chunks={chunk_index}/{len(tasks)} "
                    f"last={returned[0]['algorithm']}/{returned[0]['benchmark']} "
                    f"elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
    report = validate_and_export(
        args.output_root, mappings, runs, max_fes, mode
    )
    print(
        json.dumps(
            {"elapsed_seconds": time.time()-started, **report},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
