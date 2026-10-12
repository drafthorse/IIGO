"""IIGO with diversity-guided inter-dimensional gap exchange (IGE).

Population diversity combines the median normalized angular volume of
centered samples with normalized RMS spread. Its sliding-window trend
controls IGE and stage-dependent dimension selection. The IGE update
probability is ``gr + 0.25 * diversity``.
"""

import math

import numba
import numpy as np

from elements import F


# ---------------------------------------------------------------------------
# Algorithm parameters
# ---------------------------------------------------------------------------

WINDOW_FRACTION = 0.30
GEOMETRIC_BASE_PROBABILITY = 0.15
DIVERSITY_PROBABILITY_WEIGHT = 0.25
BEST_ATTRACTION = 0.50

EARLY_STAGE_END = 0.30
LATE_STAGE_START = 0.80
DIVERSITY_ZERO_TOLERANCE = 1e-20
VOLUME_DENOMINATOR_TOLERANCE = 1e-30

OSCILLATION_AMPLITUDE = 0.50
OSCILLATION_DECAY = 3.0
OSCILLATION_CYCLES = 5.0
OSCILLATION_PHASE = 0.5 * math.pi


@numba.njit(nogil=True)
def _clip_to_bounds(candidate, lower, upper):
    for coordinate in range(candidate.shape[0]):
        if candidate[coordinate] < lower:
            candidate[coordinate] = lower
        elif candidate[coordinate] > upper:
            candidate[coordinate] = upper
    return candidate


@numba.njit(nogil=True)
def _evaluate_population(population, matrix, shift, function_number, shuffle,
                         benchmark_id):
    return F(
        population, matrix, shift, function_number, shuffle, benchmark_id
    )[:, 0]


@numba.njit(nogil=True)
def _evaluate_one(candidate, matrix, shift, function_number, shuffle,
                  benchmark_id):
    return F(
        candidate.reshape((1, candidate.shape[0])),
        matrix,
        shift,
        function_number,
        shuffle,
        benchmark_id,
    )[0, 0]


@numba.njit(nogil=True)
def _initialize(max_fes, initial_population, matrix, shift, function_number,
                shuffle, benchmark_id):
    population = initial_population.copy()
    fitness = _evaluate_population(
        population, matrix, shift, function_number, shuffle, benchmark_id
    )
    evaluations = population.shape[0]
    best_index = int(np.argmin(fitness))
    best_fitness = fitness[best_index]
    best_position = population[best_index].copy()

    convergence = np.empty(max_fes + 1, dtype=np.float64)
    evaluation_axis = np.empty(max_fes + 1, dtype=np.int64)
    convergence[0] = best_fitness
    evaluation_axis[0] = evaluations
    return (
        population,
        fitness,
        evaluations,
        best_position,
        best_fitness,
        convergence,
        evaluation_axis,
    )


@numba.njit(nogil=True)
def _population_mean(population):
    mean = np.zeros(population.shape[1])
    for individual in range(population.shape[0]):
        mean += population[individual]
    return mean / population.shape[0]


@numba.njit(nogil=True)
def _centered_volume_diversity(population):
    """Return the median angular-volume component from NP random samples.

    Using the current population size as the number of samples avoids adding a
    separately tuned sampling parameter.  Every sample uses three distinct
    individuals and three distinct dimensions.  Centering removes dependence
    on the coordinate origin; the normalized determinant remains in [0, 1]
    up to floating-point round-off by Hadamard's inequality.
    """
    population_size, dimension = population.shape
    if population_size < 3 or dimension < 3:
        return 0.0

    center = _population_mean(population)
    volumes = np.empty(population_size)

    for sample in range(population_size):
        individual_0 = np.random.randint(population_size)
        individual_1 = np.random.randint(population_size)
        while individual_1 == individual_0:
            individual_1 = np.random.randint(population_size)
        individual_2 = np.random.randint(population_size)
        while (individual_2 == individual_0
               or individual_2 == individual_1):
            individual_2 = np.random.randint(population_size)

        dimension_0 = np.random.randint(dimension)
        dimension_1 = np.random.randint(dimension)
        while dimension_1 == dimension_0:
            dimension_1 = np.random.randint(dimension)
        dimension_2 = np.random.randint(dimension)
        while dimension_2 == dimension_0 or dimension_2 == dimension_1:
            dimension_2 = np.random.randint(dimension)

        a00 = population[individual_0, dimension_0] - center[dimension_0]
        a01 = population[individual_0, dimension_1] - center[dimension_1]
        a02 = population[individual_0, dimension_2] - center[dimension_2]
        a10 = population[individual_1, dimension_0] - center[dimension_0]
        a11 = population[individual_1, dimension_1] - center[dimension_1]
        a12 = population[individual_1, dimension_2] - center[dimension_2]
        a20 = population[individual_2, dimension_0] - center[dimension_0]
        a21 = population[individual_2, dimension_1] - center[dimension_1]
        a22 = population[individual_2, dimension_2] - center[dimension_2]

        determinant = (
            a00 * (a11 * a22 - a12 * a21)
            - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20)
        )
        norm_0 = math.sqrt(a00 * a00 + a01 * a01 + a02 * a02)
        norm_1 = math.sqrt(a10 * a10 + a11 * a11 + a12 * a12)
        norm_2 = math.sqrt(a20 * a20 + a21 * a21 + a22 * a22)
        denominator = norm_0 * norm_1 * norm_2

        if denominator <= VOLUME_DENOMINATOR_TOLERANCE:
            volumes[sample] = 0.0
        else:
            value = abs(determinant) / denominator
            volumes[sample] = min(1.0, max(0.0, value))

    return np.median(volumes)


@numba.njit(nogil=True)
def _normalized_rms_spread(population, lower, upper):
    """Scale component linked to the trace of the population covariance.

    For a common box range R=upper-lower, this is

        2/R * sqrt(trace(Sigma)/D),

    where Sigma uses the population divisor.  Popoviciu's variance bound makes
    the component lie in [0, 1] for a population inside the common box.
    """
    center = _population_mean(population)
    squared_sum = 0.0
    for individual in range(population.shape[0]):
        for coordinate in range(population.shape[1]):
            difference = population[individual, coordinate] - center[coordinate]
            squared_sum += difference * difference

    mean_square = squared_sum / (
        population.shape[0] * population.shape[1]
    )
    box_range = upper - lower
    if box_range <= 0.0:
        return 0.0
    spread = 2.0 * math.sqrt(mean_square) / box_range
    return min(1.0, max(0.0, spread))


@numba.njit(nogil=True)
def _population_diversity(population, lower, upper):
    """Bounded shape-scale diversity used by the sliding window.

    The product contains no mixing weight: the median normalized determinant
    measures random-subspace angular diversity, while normalized RMS spread
    supplies the missing scale sensitivity.  Both factors are translation
    invariant and bounded in [0, 1].
    """
    angular_volume = _centered_volume_diversity(population)
    spread = _normalized_rms_spread(population, lower, upper)
    return angular_volume * spread


@numba.njit(nogil=True)
def _trend_slope(window):
    """Least-squares slope of diversity values in the sliding window."""
    size = window.shape[0]
    if size < 2:
        return 0.0

    x_mean = 0.5 * (size - 1)
    y_mean = 0.0
    for index in range(size):
        y_mean += window[index]
    y_mean /= size

    numerator = 0.0
    denominator = 0.0
    for index in range(size):
        dx = index - x_mean
        numerator += dx * (window[index] - y_mean)
        denominator += dx * dx
    return 0.0 if denominator == 0.0 else numerator / denominator


@numba.njit(nogil=True)
def _enqueue_diversity(window, count, diversity):
    """FIFO enqueue followed by a slope update when the window is full."""
    if count < window.shape[0]:
        window[count] = diversity
        count += 1
    else:
        for index in range(window.shape[0] - 1):
            window[index] = window[index + 1]
        window[-1] = diversity

    trend = 1.0
    if count == window.shape[0]:
        trend = _trend_slope(window)
    return count, trend


@numba.njit(nogil=True)
def _decay_oscillation(total_generations, generation):
    """Positive part of the decaying oscillation function."""
    amplitude = OSCILLATION_AMPLITUDE
    exponent = -OSCILLATION_DECAY * generation / total_generations
    angle = (
        2.0 * math.pi * OSCILLATION_CYCLES
        * generation / total_generations
        + OSCILLATION_PHASE
    )
    value = amplitude * math.exp(exponent) * math.sin(angle)
    return max(0.0, value)


@numba.njit(nogil=True)
def _ige_trial(population, target_index, best_index, diversity):
    """Construct an IGE trial using sibling gaps and cosine modulation."""
    population_size, dimension = population.shape
    sibling = np.random.randint(population_size)
    while sibling == target_index:
        sibling = np.random.randint(population_size)

    mutation_probability = min(
        1.0,
        GEOMETRIC_BASE_PROBABILITY
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
            horizontal_gap = (
                companion - population[sibling, direction]
            )
            vertical_gap = target - companion
            best_gap = population[best_index, coordinate] - target
            trial[coordinate] = (
                population[target_index, direction]
                + horizontal_gap
                + vertical_gap
                * math.cos(0.5 * math.pi * np.random.random())
                + BEST_ATTRACTION * best_gap
            )

    return trial


@numba.njit(nogil=True)
def _stage_selection_trial(population, target_index, best_index, diversity,
                           progress, decay, lower, upper):
    """Construct a trial according to the early, middle or late search stage."""
    dimension = population.shape[1]

    if progress > LATE_STAGE_START:
        distance = np.abs(
            population[best_index] - population[target_index]
        )
        spiral_position = 2.0 * np.random.random() - 1.0
        return (
            population[best_index]
            + distance
            * math.cos(2.0 * math.pi * spiral_position)
            * math.exp(decay * spiral_position)
        )

    if progress < EARLY_STAGE_END or diversity < DIVERSITY_ZERO_TOLERANCE:
        return np.random.random(dimension) * (upper - lower) + lower

    return population[target_index].copy()


@numba.njit(nogil=True)
def iigo_final(max_fes, search_num, lower, upper, dimension, matrix, shift,
               function_number, shuffle, benchmark_id, seed,
               initial_population):
    """Run IIGO under the specified function evaluation budget.

    ``search_num`` and ``dimension`` remain in the public signature for drop-in
    compatibility.  Population and dimension sizes are taken from the supplied
    initial population.
    """
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
    actual_dimension = population.shape[1]
    window_size = max(
        2, int(math.ceil(WINDOW_FRACTION * initial_size)) + 1
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
                trial = _ige_trial(
                    population, target_index, best_index, diversity
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


IIGO = iigo_final

__all__ = ["iigo_final", "IIGO"]
