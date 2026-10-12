"""Run sensitivity experiments, varying one IIGO parameter at a time."""

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
from IIGO_SensitivityVariants import (
    COSINE_MODULATION,
    GAUSSIAN_MODULATION,
    run_sensitivity_variant,
)
from elements import F
from experiment_config import BENCHMARK_IDS, PAPER_FUNCTIONS, derive_seed


ROOT = Path(__file__).resolve().parent
DIMENSION = 30
POPULATION_SIZE = 10
MAX_FES = 300_000
SMOKE_CASES = {"CEC2014": (30,), "CEC2017": (29,), "CEC2022": (12,)}

CONDITIONS = {
    "baseline": {
        "factor": "baseline", "setting": "default",
        "gr": 0.15, "window_fraction": 0.30,
        "modulation": COSINE_MODULATION,
    },
    "gr_0.05": {
        "factor": "gr", "setting": "0.05",
        "gr": 0.05, "window_fraction": 0.30,
        "modulation": COSINE_MODULATION,
    },
    "gr_0.10": {
        "factor": "gr", "setting": "0.10",
        "gr": 0.10, "window_fraction": 0.30,
        "modulation": COSINE_MODULATION,
    },
    "gr_0.20": {
        "factor": "gr", "setting": "0.20",
        "gr": 0.20, "window_fraction": 0.30,
        "modulation": COSINE_MODULATION,
    },
    "gr_0.25": {
        "factor": "gr", "setting": "0.25",
        "gr": 0.25, "window_fraction": 0.30,
        "modulation": COSINE_MODULATION,
    },
    "window_0.10": {
        "factor": "window_fraction", "setting": "0.10",
        "gr": 0.15, "window_fraction": 0.10,
        "modulation": COSINE_MODULATION,
    },
    "window_0.20": {
        "factor": "window_fraction", "setting": "0.20",
        "gr": 0.15, "window_fraction": 0.20,
        "modulation": COSINE_MODULATION,
    },
    "window_0.40": {
        "factor": "window_fraction", "setting": "0.40",
        "gr": 0.15, "window_fraction": 0.40,
        "modulation": COSINE_MODULATION,
    },
    "window_0.50": {
        "factor": "window_fraction", "setting": "0.50",
        "gr": 0.15, "window_fraction": 0.50,
        "modulation": COSINE_MODULATION,
    },
    "modulation_gaussian": {
        "factor": "modulation", "setting": "Gaussian",
        "gr": 0.15, "window_fraction": 0.30,
        "modulation": GAUSSIAN_MODULATION,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths(output_root: Path, condition: str, benchmark: str,
          cec_function: int, run: int) -> tuple[Path, Path]:
    paper_function = PAPER_FUNCTIONS[benchmark][cec_function]
    directory = output_root / condition / benchmark / f"F{paper_function}"
    stem = f"{condition}_F{paper_function}_{run}"
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def complete(output_root: Path, condition: str, benchmark: str,
             cec_function: int, run: int, max_fes: int) -> bool:
    result_path, metadata_path = paths(
        output_root, condition, benchmark, cec_function, run
    )
    if not result_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata["final_fes"]) != max_fes:
            return False
        if int(metadata["seed"]) != derive_seed(benchmark, cec_function, run):
            return False
        if int(metadata["population_size"]) != POPULATION_SIZE:
            return False
        tolerance = 1e-8 * (1.0 + abs(float(metadata["best_fitness"])))
        if float(metadata["fitness_position_abs_error"]) > tolerance:
            return False
        with np.load(result_path) as payload:
            position = np.asarray(payload["best_position"])
            stored = float(np.asarray(payload["best_fitness"]).reshape(-1)[0])
        return (
            position.shape == (DIMENSION,)
            and np.isfinite(position).all()
            and stored == float(metadata["best_fitness"])
        )
    except Exception:
        return False


def run_one(output_root: str, condition: str, benchmark: str,
            cec_function: int, run: int, max_fes: int) -> dict:
    config = CONDITIONS[condition]
    matrix, shift, shuffle = Interface.load_cec_input(
        benchmark, cec_function, DIMENSION
    )
    seed, initial_population, initial_hash = Interface.make_initial_population(
        benchmark, cec_function, run, POPULATION_SIZE, DIMENSION
    )
    curve, fes, best_position, best_fitness = run_sensitivity_variant(
        max_fes,
        POPULATION_SIZE,
        -100.0,
        100.0,
        DIMENSION,
        matrix,
        shift,
        cec_function,
        shuffle,
        BENCHMARK_IDS[benchmark],
        seed,
        initial_population,
        config["gr"],
        config["window_fraction"],
        config["modulation"],
    )
    reevaluated = float(
        F(
            np.ascontiguousarray(best_position.reshape(1, -1)),
            matrix,
            shift,
            cec_function,
            shuffle,
            BENCHMARK_IDS[benchmark],
        )[0, 0]
    )
    result_path, metadata_path = paths(
        Path(output_root), condition, benchmark, cec_function, run
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        result_path,
        best_position=np.asarray(best_position),
        best_fitness=np.asarray([best_fitness]),
    )
    metadata = {
        "condition": condition,
        "factor": config["factor"],
        "setting": config["setting"],
        "gr": config["gr"],
        "window_fraction": config["window_fraction"],
        "modulation": (
            "cosine" if config["modulation"] == COSINE_MODULATION
            else "Gaussian"
        ),
        "benchmark": benchmark,
        "benchmark_id": BENCHMARK_IDS[benchmark],
        "cec_function": int(cec_function),
        "paper_function": int(PAPER_FUNCTIONS[benchmark][cec_function]),
        "run": int(run),
        "dimension": DIMENSION,
        "population_size": POPULATION_SIZE,
        "seed": int(seed),
        "initial_population_sha256": initial_hash,
        "final_fes": int(fes[-1]),
        "curve_monotonic": bool(
            curve.size < 2
            or np.all(
                np.diff(curve)
                <= 1e-10 * np.maximum(1.0, np.abs(curve[:-1]))
            )
        ),
        "best_fitness": float(best_fitness),
        "reevaluated_best_fitness": reevaluated,
        "fitness_position_abs_error": float(
            abs(reevaluated - best_fitness)
        ),
        "result_path": str(result_path.resolve()),
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, metadata_path)
    return metadata


def run_chunk(task: tuple) -> list[dict]:
    output_root, condition, benchmark, cec_function, runs, max_fes = task
    return [
        run_one(
            output_root, condition, benchmark, cec_function, run, max_fes
        )
        for run in runs
    ]


def build_tasks(output_root: Path, mappings: dict, runs: int, max_fes: int,
                chunk_size: int, conditions: tuple[str, ...]
                ) -> tuple[list[tuple], int]:
    tasks = []
    already_complete = 0
    for condition in conditions:
        for benchmark, mapping in mappings.items():
            functions = tuple(mapping)
            for cec_function in functions:
                pending = []
                for run in range(runs):
                    if complete(
                        output_root, condition, benchmark, cec_function,
                        run, max_fes
                    ):
                        already_complete += 1
                    else:
                        pending.append(run)
                for start in range(0, len(pending), chunk_size):
                    tasks.append((
                        str(output_root),
                        condition,
                        benchmark,
                        cec_function,
                        tuple(pending[start:start + chunk_size]),
                        max_fes,
                    ))
    return tasks, already_complete


def write_manifest(output_root: Path, mappings: dict, runs: int,
                   max_fes: int, mode: str,
                   conditions: tuple[str, ...]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    source_files = (
        "IIGO_Final.py",
        "IIGO_SensitivityVariants.py",
        "run_iigo_parameter_sensitivity.py",
        "Interface.py",
        "experiment_config.py",
        "elements.py",
    )
    manifest = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run+validate",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "verification_status": "PENDING",
            "version_label": f"iigo_parameter_sensitivity_{mode}_v1",
        },
        "design": "one factor at a time",
        "baseline": {
            "population_size": POPULATION_SIZE,
            "gr": 0.15,
            "window_fraction": 0.30,
            "modulation": "cosine",
        },
        "conditions": {name: CONDITIONS[name] for name in conditions},
        "dimension": DIMENSION,
        "max_fes_actual_objective_calls": max_fes,
        "independent_runs": runs,
        "function_mapping": mappings,
        "initialization_policy": (
            "All conditions receive the same seed and byte-identical initial "
            "population within each suite/function/run block."
        ),
        "gaussian_control": (
            "min(1, abs(N(0,1))) replaces only the cosine multiplier. "
            "It retains support [0,1] and approximately matches the cosine "
            "multiplier's expected magnitude."
        ),
        "source_sha256": {
            name: sha256(ROOT / name) for name in source_files
        },
    }
    (output_root / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def validate_and_export(output_root: Path, mappings: dict, runs: int,
                        max_fes: int, mode: str,
                        conditions: tuple[str, ...]) -> dict:
    failures = []
    rows = []
    paired = {}
    expected = len(conditions) * sum(len(x) for x in mappings.values()) * runs
    for condition in conditions:
        for benchmark, mapping in mappings.items():
            for cec_function in tuple(mapping):
                for run in range(runs):
                    key = f"{condition}/{benchmark}/F{cec_function}/run{run}"
                    result_path, metadata_path = paths(
                        output_root, condition, benchmark, cec_function, run
                    )
                    if not result_path.exists() or not metadata_path.exists():
                        failures.append(f"missing:{key}")
                        continue
                    metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    if int(metadata["final_fes"]) != max_fes:
                        failures.append(f"budget:{key}")
                    if not bool(metadata["curve_monotonic"]):
                        failures.append(f"curve:{key}")
                    if int(metadata["seed"]) != derive_seed(
                        benchmark, cec_function, run
                    ):
                        failures.append(f"seed:{key}")
                    if float(metadata["fitness_position_abs_error"]) > (
                        1e-8 * (1.0 + abs(float(metadata["best_fitness"])))
                    ):
                        failures.append(f"position_fitness:{key}")
                    pair_key = (benchmark, cec_function, run)
                    item = paired.setdefault(
                        pair_key,
                        {"seeds": set(), "hashes": set(), "conditions": set()},
                    )
                    item["seeds"].add(int(metadata["seed"]))
                    item["hashes"].add(metadata["initial_population_sha256"])
                    item["conditions"].add(condition)
                    rows.append(metadata)
    for key, item in paired.items():
        if (
            len(item["seeds"]) != 1
            or len(item["hashes"]) != 1
            or item["conditions"] != set(conditions)
        ):
            failures.append(f"pairing:{key}")

    rows.sort(key=lambda x: (x["condition"], x["paper_function"], x["run"]))
    if rows:
        fieldnames = [
            "condition", "factor", "setting", "gr", "window_fraction",
            "modulation", "benchmark", "benchmark_id", "cec_function",
            "paper_function",
            "run", "dimension", "population_size", "seed", "final_fes",
            "best_fitness", "reevaluated_best_fitness",
            "fitness_position_abs_error", "initial_population_sha256",
            "curve_monotonic", "result_path",
        ]
        with (output_root / "full_results.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        grouped = {}
        for row in rows:
            group_key = (
                row["condition"], row["factor"], row["setting"],
                row["paper_function"], row["benchmark"], row["cec_function"],
            )
            grouped.setdefault(group_key, []).append(row["best_fitness"])
        summary = []
        for group_key, values in sorted(
            grouped.items(), key=lambda x: (x[0][0], x[0][3])
        ):
            values = np.asarray(values, dtype=np.float64)
            summary.append({
                "condition": group_key[0],
                "factor": group_key[1],
                "setting": group_key[2],
                "paper_function": group_key[3],
                "benchmark": group_key[4],
                "cec_function": group_key[5],
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
            "version_label": f"iigo_parameter_sensitivity_{mode}_v1",
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
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data4" / "iigo_final_parameter_sensitivity"
        / "full_30d_300000fes",
    )
    args = parser.parse_args()
    mappings = SMOKE_CASES if args.smoke else PAPER_FUNCTIONS
    mode = "smoke" if args.smoke else "full"
    runs = 2 if args.smoke and args.runs == 30 else args.runs
    max_fes = 5_000 if args.smoke and args.max_fes == MAX_FES else args.max_fes
    conditions = tuple(args.conditions or CONDITIONS)
    if args.workers < 1 or args.chunk_size < 1 or runs < 1:
        raise SystemExit("workers, chunk-size and runs must be positive")
    if max_fes < POPULATION_SIZE:
        raise SystemExit("max-fes must cover initialization")

    write_manifest(
        args.output_root, mappings, runs, max_fes, mode, conditions
    )
    tasks, already_complete = build_tasks(
        args.output_root, mappings, runs, max_fes, args.chunk_size, conditions
    )
    total = len(conditions) * sum(len(x) for x in mappings.values()) * runs
    print(
        f"IIGO parameter sensitivity ({mode}): conditions={conditions}, "
        f"target={total}, already_complete={already_complete}, "
        f"chunks={len(tasks)}",
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
                    f"progress={completed}/{total} "
                    f"chunks={chunk_index}/{len(tasks)} "
                    f"last={returned[0]['condition']}/"
                    f"{returned[0]['benchmark']}/F"
                    f"{returned[0]['paper_function']} "
                    f"elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
    report = validate_and_export(
        args.output_root, mappings, runs, max_fes, mode, conditions
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
