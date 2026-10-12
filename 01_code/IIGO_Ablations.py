"""IIGO ablation variants that remove one mechanism at a time.

This module is experiment-only.  It does not replace the public ``IIGO``
entry and it does not modify :mod:`IIGO_Final`.

Variant definitions:

``IIGO_NoWindow``
    Replaces the multi-generation FIFO trend with the signed one-generation
    diversity change.  The same diversity metric and both trial operators are
    retained.

``IIGO_NoIGE``
    Replaces every IGE-branch trial by an unchanged target.  The controller,
    diversity metric and stage-selection branch are retained.  The unchanged
    target is still evaluated so all variants use the same FE budget.

``IIGO_NoStage``
    Replaces the early/middle/late stage-dependent trial by the early-stage
    uniform exploration trial at every progress value.  The controller, IGE
    branch and diversity metric are retained.
"""

import math

import numba
import numpy as np

from IIGO_Final import (
    WINDOW_FRACTION,
    _clip_to_bounds,
    _decay_oscillation,
    _enqueue_diversity,
    _evaluate_one,
    _ige_trial,
    _initialize,
    _population_diversity,
    _stage_selection_trial,
    iigo_final,
)


NO_WINDOW = 1
NO_IGE = 2
NO_STAGE = 3


@numba.njit(nogil=True)
def _uniform_stage_neutral_trial(population, lower, upper):
    """Progress-neutral replacement for the stage-dependent trial."""
    return np.random.random(population.shape[1]) * (upper - lower) + lower


@numba.njit(nogil=True)
def _run_ablation(max_fes, search_num, lower, upper, dimension, matrix, shift,
                  function_number, shuffle, benchmark_id, seed,
                  initial_population, mode):
    np.random.seed(seed)
    (
        population,
        fitness,
        evaluations,
        best_position,
        best_fitness,
        convergence,
        evaluation_axis,
    ) = _initialize(
        max_fes,
        initial_population,
        matrix,
        shift,
        function_number,
        shuffle,
        benchmark_id,
    )

    initial_size = population.shape[0]
    window_size = max(2, int(math.ceil(WINDOW_FRACTION * initial_size)) + 1)
    diversity_window = np.empty(window_size)
    window_count = 0

    diversity = _population_diversity(population, lower, upper)
    previous_diversity = diversity
    trend = 1.0
    if mode != NO_WINDOW:
        window_count, trend = _enqueue_diversity(
            diversity_window, window_count, diversity
        )

    remaining_after_initialization = max(0, max_fes - evaluations)
    total_generations = max(
        1, int(math.ceil(remaining_after_initialization / initial_size))
    )
    generation = 1
    record = 1

    while evaluations < max_fes:
        population_size = population.shape[0]
        progress = min(1.0, generation / total_generations)
        decay = _decay_oscillation(total_generations, generation)
        best_index = int(np.argmin(fitness))
        batch_size = min(population_size, max_fes - evaluations)
        next_population = population.copy()

        for target_index in range(batch_size):
            use_ige_branch = trend > 0.0 or diversity >= 1.0 - 1e-15
            if use_ige_branch:
                if mode == NO_IGE:
                    trial = population[target_index].copy()
                else:
                    trial = _ige_trial(
                        population, target_index, best_index, diversity
                    )
            else:
                if mode == NO_STAGE:
                    trial = _uniform_stage_neutral_trial(
                        population, lower, upper
                    )
                else:
                    trial = _stage_selection_trial(
                        population,
                        target_index,
                        best_index,
                        diversity,
                        progress,
                        decay,
                        lower,
                        upper,
                    )

                if mode == NO_WINDOW:
                    trend = 1.0
                else:
                    diversity_window[:] = 0.0
                    window_count = 0
                    trend = 1.0

            trial = _clip_to_bounds(trial, lower, upper)
            trial_fitness = _evaluate_one(
                trial,
                matrix,
                shift,
                function_number,
                shuffle,
                benchmark_id,
            )
            if trial_fitness <= fitness[target_index]:
                next_population[target_index] = trial
                fitness[target_index] = trial_fitness

        population = next_population
        evaluations += batch_size

        best_index = int(np.argmin(fitness))
        if fitness[best_index] < best_fitness:
            best_fitness = fitness[best_index]
            best_position = population[best_index].copy()

        diversity = _population_diversity(population, lower, upper)
        if mode == NO_WINDOW:
            trend = diversity - previous_diversity
            previous_diversity = diversity
        else:
            window_count, trend = _enqueue_diversity(
                diversity_window, window_count, diversity
            )

        convergence[record] = best_fitness
        evaluation_axis[record] = evaluations
        record += 1
        generation += 1

    return (
        convergence[:record],
        evaluation_axis[:record],
        best_position,
        best_fitness,
    )


def iigo_no_window(max_fes, search_num, lower, upper, dimension, matrix,
                   shift, function_number, shuffle, benchmark_id, seed,
                   initial_population):
    return _run_ablation(
        max_fes, search_num, lower, upper, dimension, matrix, shift,
        function_number, shuffle, benchmark_id, seed, initial_population,
        NO_WINDOW,
    )


def iigo_no_ige(max_fes, search_num, lower, upper, dimension, matrix, shift,
                function_number, shuffle, benchmark_id, seed,
                initial_population):
    return _run_ablation(
        max_fes, search_num, lower, upper, dimension, matrix, shift,
        function_number, shuffle, benchmark_id, seed, initial_population,
        NO_IGE,
    )


def iigo_no_stage(max_fes, search_num, lower, upper, dimension, matrix, shift,
                  function_number, shuffle, benchmark_id, seed,
                  initial_population):
    return _run_ablation(
        max_fes, search_num, lower, upper, dimension, matrix, shift,
        function_number, shuffle, benchmark_id, seed, initial_population,
        NO_STAGE,
    )


ABLATION_FUNCTIONS = {
    "IIGO_Full": iigo_final,
    "IIGO_NoWindow": iigo_no_window,
    "IIGO_NoIGE": iigo_no_ige,
    "IIGO_NoStage": iigo_no_stage,
}

__all__ = [
    "ABLATION_FUNCTIONS",
    "iigo_no_window",
    "iigo_no_ige",
    "iigo_no_stage",
]
