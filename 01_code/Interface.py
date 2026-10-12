"""Input loading, deterministic initialization and result persistence."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

import ExperimentAlgorithms as algorithms
from elements import F
from experiment_config import BENCHMARK_IDS, PAPER_FUNCTIONS, derive_seed


def load_cec_input(benchmark, function_number, dimension, input_root="input"):
    root = Path(input_root) / f"{benchmark}_input_data"
    matrix = np.loadtxt(root / f"M_{function_number}_D{dimension}.txt")
    if benchmark == "CEC2022":
        shift = np.loadtxt(root / f"shift_data_{function_number}.txt")
        shuffle_function = function_number if 7 <= function_number <= 8 else 6
    else:
        shift = np.loadtxt(root / f"shift_data_{function_number}.txt")
        shuffle_function = function_number
    shuffle = np.loadtxt(
        root / f"shuffle_data_{shuffle_function}_D{dimension}.txt",
        dtype=np.int32,
    )
    return (np.ascontiguousarray(matrix, dtype=np.float64),
            np.ascontiguousarray(shift.reshape(-1), dtype=np.float64),
            np.ascontiguousarray(shuffle.reshape(-1)-1, dtype=np.int32))


def make_initial_population(benchmark, function_number, run_number,
                            population_size, dimension, lower=-100.0, upper=100.0):
    seed = derive_seed(benchmark, function_number, run_number)
    rng = np.random.default_rng(seed)
    population = rng.uniform(lower, upper, size=(population_size, dimension))
    population = np.ascontiguousarray(population, dtype=np.float64)
    digest = hashlib.sha256(population.tobytes(order="C")).hexdigest()
    return seed, population, digest


def run_one(dimension, population_size, run_number, function_number, optimizer,
            benchmark, max_fes, output_root):
    matrix, shift, shuffle = load_cec_input(benchmark, function_number, dimension)
    seed, initial_population, initial_hash = make_initial_population(
        benchmark, function_number, run_number, population_size, dimension)
    function = algorithms.get_f(optimizer)
    curve, fe_curve, best_position, best_fitness = function(
        max_fes, population_size, -100.0, 100.0, dimension,
        matrix, shift, function_number, shuffle,
        BENCHMARK_IDS[benchmark], seed, initial_population,
    )

    # One post-run diagnostic call is deliberately outside the optimization
    # budget. It checks that the returned individual and fitness match.
    reevaluated = float(F(
        np.ascontiguousarray(best_position.reshape(1, -1)), matrix, shift,
        function_number, shuffle, BENCHMARK_IDS[benchmark])[0, 0])

    paper_number = PAPER_FUNCTIONS[benchmark][function_number]
    directory = Path(output_root) / optimizer / benchmark / f"F{paper_number}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{optimizer}_F{paper_number}_{run_number}"
    result_path = directory / f"{stem}.npz"
    metadata_path = directory / f"{stem}.json"
    np.savez_compressed(
        result_path,
        convergence=np.asarray(curve),
        fes=np.asarray(fe_curve),
        best_position=np.asarray(best_position),
        best_fitness=np.asarray([best_fitness]),
    )
    metadata = {
        "optimizer": optimizer,
        "benchmark": benchmark,
        "benchmark_id": BENCHMARK_IDS[benchmark],
        "cec_function": int(function_number),
        "paper_function": int(paper_number),
        "run_number": int(run_number),
        "dimension": int(dimension),
        "population_size": int(population_size),
        "seed": int(seed),
        "initial_population_sha256": initial_hash,
        "optimization_fes": int(fe_curve[-1]),
        "diagnostic_fes": 1,
        "best_fitness": float(best_fitness),
        "reevaluated_best_fitness": reevaluated,
        "fitness_position_abs_error": float(abs(reevaluated-best_fitness)),
        "result_path": str(result_path.resolve()),
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, metadata_path)
    return metadata


def run_pair(args):
    (dimension, population_size, function_number, optimizer, benchmark,
     max_fes, output_root, run_numbers) = args
    return [run_one(
        dimension, population_size, run_number, function_number, optimizer,
        benchmark, max_fes, output_root,
    ) for run_number in run_numbers]
