"""Convenient entry points for running project optimizers on engineering cases."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np

import EngineerBenchmark
from ExperimentAlgorithms import get_f


DEFAULT_ENGINEERING_MAX_FES = 300_000
DEFAULT_ENGINEERING_POPULATION = 10
TRUSS_RECOMMENDED_MAX_FES = {
    "TRUSS17": 170_000,
    "TRUSS18": 120_000,
    "TRUSS25": 80_000,
    "TRUSS72": 160_000,
    "TRUSS120": 70_000,
    "TRUSS200": 290_000,
}


def engineering_seed(problem: Union[str, int], run_number: int = 0) -> int:
    """Return a stable seed independent of Python's randomized string hashes."""
    problem_id = EngineerBenchmark.benchmark_to_number(problem)
    return int((9_000_001 + 10_007 * problem_id + 1_000_003 * run_number) % 2_147_483_647)


def make_initial_population(
    problem: Union[str, int],
    population_size: int = DEFAULT_ENGINEERING_POPULATION,
    *,
    seed: Optional[int] = None,
    run_number: int = 0,
) -> np.ndarray:
    """Create a reproducible population in the normalized search space."""
    if population_size < 4:
        raise ValueError("population_size must be at least 4 for the project algorithms")
    if seed is None:
        seed = engineering_seed(problem, run_number)
    dimension = EngineerBenchmark.problem_dimension(problem)
    return np.random.default_rng(seed).uniform(0.0, 1.0, (population_size, dimension))


def run_engineering_case(
    algorithm: str,
    problem: Union[str, int],
    *,
    max_fes: Optional[int] = None,
    population_size: int = DEFAULT_ENGINEERING_POPULATION,
    run_number: int = 0,
    seed: Optional[int] = None,
    initial_population: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Run one current optimizer and return both optimizer and physical results.

    The optimizer sees only normalized variables.  ``physical_solution`` and
    ``objective`` in the returned dictionary are ready for engineering tables.
    """
    problem_id = EngineerBenchmark.benchmark_to_number(problem)
    canonical_name = EngineerBenchmark.problem_name(problem_id)
    dimension = EngineerBenchmark.problem_dimension(problem_id)
    if max_fes is None:
        max_fes = TRUSS_RECOMMENDED_MAX_FES.get(
            canonical_name, DEFAULT_ENGINEERING_MAX_FES
        )
    if max_fes < population_size:
        raise ValueError("max_fes must be no smaller than population_size")
    if seed is None:
        seed = engineering_seed(problem_id, run_number)

    if initial_population is None:
        initial_pop = make_initial_population(
            problem_id, population_size, seed=seed, run_number=run_number
        )
    else:
        initial_pop = np.asarray(initial_population, dtype=np.float64).copy()
        expected_shape = (population_size, dimension)
        if initial_pop.shape != expected_shape:
            raise ValueError(
                f"initial_population shape is {initial_pop.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(initial_pop)):
            raise ValueError("initial_population contains a non-finite value")
        np.clip(initial_pop, 0.0, 1.0, out=initial_pop)

    # CEC-only inputs are intentionally inert for engineering dispatch.
    dummy_rotation = np.zeros((1, 1), dtype=np.float64)
    dummy_shift = np.zeros(1, dtype=np.float64)
    dummy_shuffle = np.zeros(1, dtype=np.int32)

    optimizer = get_f(algorithm)
    curve, fe_curve, normalized_best, optimizer_merit = optimizer(
        int(max_fes),
        int(population_size),
        0.0,
        1.0,
        int(dimension),
        dummy_rotation,
        dummy_shift,
        int(problem_id),
        dummy_shuffle,
        int(EngineerBenchmark.ENGINEERING_BENCHMARK_ID),
        int(seed),
        np.ascontiguousarray(initial_pop),
    )

    raw_normalized_best = np.asarray(normalized_best, dtype=np.float64)
    normalized_best = np.clip(raw_normalized_best, 0.0, 1.0)
    returned_point_was_repaired = not np.array_equal(
        raw_normalized_best, normalized_best
    )
    physical_best = EngineerBenchmark.decode_candidate(problem_id, normalized_best)
    details = EngineerBenchmark.evaluate_solution(problem_id, physical_best)
    if not np.isclose(float(optimizer_merit), float(details["merit"]), rtol=1e-10, atol=1e-8):
        raise RuntimeError("Optimizer result does not match engineering re-evaluation")

    result: Dict[str, object] = dict(details)
    result.update(
        {
            "algorithm": algorithm,
            "problem": canonical_name,
            "seed": int(seed),
            "run_number": int(run_number),
            "population_size": int(population_size),
            "max_fes": int(max_fes),
            "normalized_solution": normalized_best,
            "optimizer_raw_normalized_solution": raw_normalized_best,
            "returned_point_was_repaired": returned_point_was_repaired,
            "optimizer_merit": float(optimizer_merit),
            "convergence_curve": np.asarray(curve, dtype=np.float64),
            "evaluation_curve": np.asarray(fe_curve, dtype=np.int64),
        }
    )
    return result


def run_engineering_suite(
    algorithms: Sequence[str],
    problems: Iterable[Union[str, int]] = ("WBD", "CBD", "SRD", "RSLD", "ELD"),
    **run_options: object,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Run a compact algorithm-by-problem suite using identical run options."""
    output: Dict[str, Dict[str, Dict[str, object]]] = {}
    for algorithm in algorithms:
        output[algorithm] = {}
        for problem in problems:
            name = EngineerBenchmark.problem_name(problem)
            output[algorithm][name] = run_engineering_case(
                algorithm, problem, **run_options
            )
    return output
