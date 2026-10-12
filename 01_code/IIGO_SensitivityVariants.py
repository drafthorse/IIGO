"""IIGO variants for controlled parameter sensitivity experiments.

The public algorithm in :mod:`IIGO_Final` is not modified.  Each variant
changes one factor while all other constants and operators remain fixed.
"""

import math

import numba
import numpy as np

from IIGO_Final import (
    BEST_ATTRACTION,
    DIVERSITY_PROBABILITY_WEIGHT,
    _clip_to_bounds,
    _decay_oscillation,
    _enqueue_diversity,
    _evaluate_one,
    _initialize,
    _population_diversity,
    _stage_selection_trial,
)


COSINE_MODULATION = 0
GAUSSIAN_MODULATION = 1


@numba.njit(nogil=True)
def _ige_trial_parameterized(population, target_index, best_index, diversity,
                             geometric_base_probability, modulation):
    population_size, dimension = population.shape
    sibling = np.random.randint(population_size)
    while sibling == target_index:
        sibling = np.random.randint(population_size)

    mutation_probability = min(
        1.0,
        geometric_base_probability
        + DIVERSITY_PROBABILITY_WEIGHT * diversity,
    )
    trial = population[target_index].copy()

    for coordinate in range(dimension):
        direction = np.random.randint(dimension)
        while direction == coordinate:
            direction = np.random.randint(dimension)

        if np.random.random() < mutation_probability:
            target = population[target_index, coordinate]
            companion = population[sibling, coordinate]
            horizontal_gap = companion - population[sibling, direction]
            vertical_gap = target - companion
            best_gap = population[best_index, coordinate] - target

            if modulation == COSINE_MODULATION:
                multiplier = math.cos(
                    0.5 * math.pi * np.random.random()
                )
            else:
                # Bounded half-normal modulation.  Clipping preserves the
                # [0, 1] support of the cosine multiplier, while its expected
                # magnitude is close to 2/pi, the cosine multiplier's mean.
                multiplier = min(1.0, abs(np.random.normal(0.0, 1.0)))

            trial[coordinate] = (
                population[target_index, direction]
                + horizontal_gap
                + vertical_gap * multiplier
                + BEST_ATTRACTION * best_gap
            )

    return trial


@numba.njit(nogil=True)
def _run_parameterized(max_fes, search_num, lower, upper, dimension, matrix,
                       shift, function_number, shuffle, benchmark_id, seed,
                       initial_population, geometric_base_probability,
                       window_fraction, modulation):
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
    window_size = max(
        2, int(math.ceil(window_fraction * initial_size)) + 1
    )
    diversity_window = np.empty(window_size)
    window_count = 0

    diversity = _population_diversity(population, lower, upper)
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
            if trend > 0.0 or diversity >= 1.0 - 1e-15:
                trial = _ige_trial_parameterized(
                    population,
                    target_index,
                    best_index,
                    diversity,
                    geometric_base_probability,
                    modulation,
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


def run_sensitivity_variant(max_fes, search_num, lower, upper, dimension,
                            matrix, shift, function_number, shuffle,
                            benchmark_id, seed, initial_population,
                            geometric_base_probability=0.15,
                            window_fraction=0.30,
                            modulation=COSINE_MODULATION):
    return _run_parameterized(
        max_fes,
        search_num,
        lower,
        upper,
        dimension,
        matrix,
        shift,
        function_number,
        shuffle,
        benchmark_id,
        seed,
        initial_population,
        geometric_base_probability,
        window_fraction,
        modulation,
    )


__all__ = [
    "COSINE_MODULATION",
    "GAUSSIAN_MODULATION",
    "run_sensitivity_variant",
]
