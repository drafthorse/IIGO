"""Static truss benchmarks translated from the supplied MATLAB suite.

The public functions reproduce the objective, constraint aggregation and
1e8 additive penalty used by the original MATLAB analysis files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

from Truss120Benchmark import truss120_details


_ARCHIVE = np.load(Path(__file__).with_name("truss_static_models.npz"))


def _array(name: str, dtype=None) -> np.ndarray:
    value = _ARCHIVE[name]
    if dtype is not None:
        value = value.astype(dtype)
    return np.ascontiguousarray(value)


def _prepare_fixed(
    nodes: np.ndarray,
    elements: np.ndarray,
    supports: np.ndarray,
    groups: np.ndarray,
    elastic_modulus: float,
    unit_weight: float,
    loads: np.ndarray,
) -> tuple[np.ndarray, ...]:
    ndof = nodes.shape[0] * 3
    fixed = supports.reshape(-1).astype(bool)
    free_global = np.flatnonzero(~fixed).astype(np.int64)
    global_to_free = np.full(ndof, -1, dtype=np.int64)
    global_to_free[free_global] = np.arange(free_global.size, dtype=np.int64)
    group_count = int(groups.max()) + 1
    stiffness_groups = np.zeros((group_count, free_global.size, free_global.size))
    lengths = np.empty(elements.shape[0])
    directions = np.empty((elements.shape[0], 3))
    free_dofs = np.empty((elements.shape[0], 6), dtype=np.int64)
    objective_coefficients = np.zeros(group_count)

    for e, (node_a, node_b) in enumerate(elements):
        delta = nodes[node_b] - nodes[node_a]
        length = float(np.linalg.norm(delta))
        direction = delta / length
        lengths[e] = length
        directions[e] = direction
        global_dofs = np.array([
            3 * node_a, 3 * node_a + 1, 3 * node_a + 2,
            3 * node_b, 3 * node_b + 1, 3 * node_b + 2,
        ], dtype=np.int64)
        free_dofs[e] = global_to_free[global_dofs]
        vector = np.concatenate((direction, -direction))
        element_stiffness = elastic_modulus / length * np.outer(vector, vector)
        group = groups[e]
        objective_coefficients[group] += length * unit_weight
        for i in range(6):
            row = free_dofs[e, i]
            if row < 0:
                continue
            for j in range(6):
                col = free_dofs[e, j]
                if col >= 0:
                    stiffness_groups[group, row, col] += element_stiffness[i, j]

    load_cases = np.asarray(loads, dtype=np.float64)
    if load_cases.ndim == 2:
        load_cases = load_cases.reshape(1, *load_cases.shape)
    load_free = np.empty((load_cases.shape[0], free_global.size))
    for case in range(load_cases.shape[0]):
        load_free[case] = load_cases[case].reshape(-1)[free_global]
    return tuple(np.ascontiguousarray(value) for value in (
        lengths, directions, free_dofs, stiffness_groups,
        objective_coefficients, load_free,
    ))


def _loads_17() -> np.ndarray:
    load = np.zeros((9, 3))
    load[8, 1] = -100_000.0
    return load


def _loads_25() -> np.ndarray:
    loads = np.zeros((2, 10, 3))
    loads[0, 0] = [0, 20_000, -5_000]
    loads[0, 1] = [0, -20_000, -5_000]
    loads[1, 0] = [1_000, 10_000, -5_000]
    loads[1, 1] = [0, 10_000, -5_000]
    loads[1, 2] = [500, 0, 0]
    loads[1, 5] = [500, 0, 0]
    return loads


def _loads_72() -> np.ndarray:
    loads = np.zeros((2, 20, 3))
    loads[0, 16] = [5_000, 5_000, -5_000]
    loads[1, 16:20, 2] = -5_000
    return loads


def _loads_200() -> np.ndarray:
    loads = np.zeros((3, 77, 3))
    horizontal_nodes = [1,6,15,20,29,34,43,48,57,62,71]
    loads[0, np.array(horizontal_nodes) - 1, 0] = 1_000.0
    vertical_nodes = [1,2,3,4,5,6,8,10,12,14,15,16,17,18,19,20,22,24,26,28,
                      29,30,31,32,33,34,36,38,40,42,43,44,45,46,47,48,50,52,
                      54,56,57,58,59,60,61,62,64,66,68,70,71,72,73,74,75]
    loads[1, np.array(vertical_nodes) - 1, 1] = -10_000.0
    loads[2] = loads[0] + loads[1]
    return loads


T17 = _prepare_fixed(_array("t17_nodes"), _array("t17_elements", np.int64),
                     _array("t17_supports", np.int64), _array("t17_groups", np.int64),
                     30_000_000.0, 0.268, _loads_17())
T25 = _prepare_fixed(_array("t25_nodes"), _array("t25_elements", np.int64),
                     _array("t25_supports", np.int64), _array("t25_groups", np.int64),
                     10_000_000.0, 0.1, _loads_25())
T72 = _prepare_fixed(_array("t72_nodes"), _array("t72_elements", np.int64),
                     _array("t72_supports", np.int64), _array("t72_groups", np.int64),
                     10_000_000.0, 0.1, _loads_72())
T200 = _prepare_fixed(_array("t200_nodes"), _array("t200_elements", np.int64),
                      _array("t200_supports", np.int64), _array("t200_groups", np.int64),
                      30_000_000.0, 0.283, _loads_200())


def _prepare_banded_contributions(stiffness_groups: np.ndarray) -> tuple[np.ndarray, ...]:
    """Store the lower triangle of grouped stiffness matrices in band form."""
    rows: list[int] = []
    offsets: list[int] = []
    groups: list[int] = []
    values: list[float] = []
    bandwidth = 0
    for group in range(stiffness_groups.shape[0]):
        matrix = stiffness_groups[group]
        nz_rows, nz_cols = np.nonzero(np.tril(matrix))
        for row, col in zip(nz_rows, nz_cols):
            offset = int(row - col)
            bandwidth = max(bandwidth, offset)
            rows.append(int(row))
            offsets.append(offset)
            groups.append(group)
            values.append(float(matrix[row, col]))
    return (
        np.ascontiguousarray(np.array(rows, dtype=np.int64)),
        np.ascontiguousarray(np.array(offsets, dtype=np.int64)),
        np.ascontiguousarray(np.array(groups, dtype=np.int64)),
        np.ascontiguousarray(np.array(values, dtype=np.float64)),
        np.array([bandwidth], dtype=np.int64),
    )


T200_BAND = _prepare_banded_contributions(T200[3])

T17_GROUPS = _array("t17_groups", np.int64)
T25_GROUPS = _array("t25_groups", np.int64)
T72_GROUPS = _array("t72_groups", np.int64)
T200_GROUPS = _array("t200_groups", np.int64)
T25_COMPRESSION = np.repeat(
    np.array([35.092, 11.590, 17.305, 35.092, 35.092, 6.759, 6.959, 11.082]) * 1000.0,
    np.array([1, 4, 4, 2, 2, 4, 4, 4]),
)

T18_ELEMENTS = _array("t18_elements", np.int64)
T18_GROUPS = _array("t18_groups", np.int64)
T18_SUPPORTS = _array("t18_supports", np.int64)
T18_FIXED = T18_SUPPORTS.reshape(-1).astype(np.bool_)
T18_FREE_GLOBAL = np.ascontiguousarray(np.flatnonzero(~T18_FIXED).astype(np.int64))
T18_GLOBAL_TO_FREE = np.full(33, -1, dtype=np.int64)
T18_GLOBAL_TO_FREE[T18_FREE_GLOBAL] = np.arange(T18_FREE_GLOBAL.size, dtype=np.int64)
T18_LOAD_FULL = np.zeros((11, 3), dtype=np.float64)
T18_LOAD_FULL[[0, 1, 3, 5, 7], 1] = -20_000.0
T18_LOAD = np.ascontiguousarray(T18_LOAD_FULL.reshape(-1)[T18_FREE_GLOBAL])


@njit(cache=True, nogil=True)
def _fixed_details(
    areas: np.ndarray,
    lengths: np.ndarray,
    directions: np.ndarray,
    free_dofs: np.ndarray,
    stiffness_groups: np.ndarray,
    objective_coefficients: np.ndarray,
    load_cases: np.ndarray,
    groups: np.ndarray,
    elastic_modulus: float,
    tensile_allowable: float,
    maximum_displacement: float,
    compression_allowable: np.ndarray,
) -> tuple[float, np.ndarray]:
    free_count = stiffness_groups.shape[1]
    stiffness = np.zeros((free_count, free_count))
    for group in range(stiffness_groups.shape[0]):
        stiffness += areas[group] * stiffness_groups[group]
    include_displacement = maximum_displacement > 0.0
    constraints_per_case = 2 if include_displacement else 1
    constraints = np.zeros(load_cases.shape[0] * constraints_per_case)

    displacements = np.linalg.solve(stiffness, load_cases.T).T
    for case in range(load_cases.shape[0]):
        displacement = displacements[case]
        stress_sum = 0.0
        for e in range(lengths.size):
            extension = 0.0
            for axis in range(3):
                dof_a = free_dofs[e, axis]
                dof_b = free_dofs[e, axis + 3]
                ua = displacement[dof_a] if dof_a >= 0 else 0.0
                ub = displacement[dof_b] if dof_b >= 0 else 0.0
                extension += directions[e, axis] * (ub - ua)
            stress = elastic_modulus * extension / lengths[e]
            allowable = tensile_allowable
            if stress < 0.0 and compression_allowable.size == lengths.size:
                allowable = compression_allowable[e]
            violation = abs(stress) / allowable - 1.0
            if violation > 0.0:
                stress_sum += violation
        constraints[case * constraints_per_case] = stress_sum
        if include_displacement:
            displacement_sum = 0.0
            for value in displacement:
                violation = abs(value) / maximum_displacement - 1.0
                if violation > 0.0:
                    displacement_sum += violation
            constraints[case * constraints_per_case + 1] = displacement_sum

    objective = 0.0
    for group in range(objective_coefficients.size):
        objective += objective_coefficients[group] * areas[group]
    return objective, constraints


@njit(cache=True, nogil=True)
def truss17_details(x: np.ndarray) -> tuple[float, np.ndarray]:
    empty = np.empty(0)
    return _fixed_details(x, T17[0], T17[1], T17[2], T17[3], T17[4], T17[5],
                          T17_GROUPS, 30_000_000.0, 50_000.0, 2.0, empty)


@njit(cache=True, nogil=True)
def truss25_details(x: np.ndarray) -> tuple[float, np.ndarray]:
    return _fixed_details(x, T25[0], T25[1], T25[2], T25[3], T25[4], T25[5],
                          T25_GROUPS, 10_000_000.0, 40_000.0, 0.35, T25_COMPRESSION)


@njit(cache=True, nogil=True)
def truss72_details(x: np.ndarray) -> tuple[float, np.ndarray]:
    empty = np.empty(0)
    return _fixed_details(x, T72[0], T72[1], T72[2], T72[3], T72[4], T72[5],
                          T72_GROUPS, 10_000_000.0, 25_000.0, 0.25, empty)


@njit(cache=True, nogil=True)
def truss200_details(x: np.ndarray) -> tuple[float, np.ndarray]:
    n = T200[3].shape[1]
    bandwidth = T200_BAND[4][0]
    lower = np.zeros((n, bandwidth + 1))
    for index in range(T200_BAND[0].size):
        row = T200_BAND[0][index]
        offset = T200_BAND[1][index]
        group = T200_BAND[2][index]
        lower[row, offset] += x[group] * T200_BAND[3][index]

    # In-place Cholesky factorization of the symmetric positive-definite band.
    for row in range(n):
        first_col = max(0, row - bandwidth)
        for col in range(first_col, row + 1):
            value = lower[row, row - col]
            first_product = max(first_col, col - bandwidth)
            for k in range(first_product, col):
                value -= lower[row, row - k] * lower[col, col - k]
            if row == col:
                lower[row, 0] = np.sqrt(value)
            else:
                lower[row, row - col] = value / lower[col, 0]

    load_cases = T200[5]
    displacements = np.empty((load_cases.shape[0], n))
    work = np.empty(n)
    for case in range(load_cases.shape[0]):
        for row in range(n):
            value = load_cases[case, row]
            first_col = max(0, row - bandwidth)
            for col in range(first_col, row):
                value -= lower[row, row - col] * work[col]
            work[row] = value / lower[row, 0]
        for row in range(n - 1, -1, -1):
            value = work[row]
            last_col = min(n - 1, row + bandwidth)
            for col in range(row + 1, last_col + 1):
                value -= lower[col, col - row] * displacements[case, col]
            displacements[case, row] = value / lower[row, 0]

    constraints = np.zeros(load_cases.shape[0])
    for case in range(load_cases.shape[0]):
        stress_sum = 0.0
        displacement = displacements[case]
        for element in range(T200[0].size):
            extension = 0.0
            for axis in range(3):
                dof_a = T200[2][element, axis]
                dof_b = T200[2][element, axis + 3]
                ua = displacement[dof_a] if dof_a >= 0 else 0.0
                ub = displacement[dof_b] if dof_b >= 0 else 0.0
                extension += T200[1][element, axis] * (ub - ua)
            stress = 30_000_000.0 * extension / T200[0][element]
            violation = abs(stress) / 10_000.0 - 1.0
            if violation > 0.0:
                stress_sum += violation
        constraints[case] = stress_sum

    objective = 0.0
    for group in range(T200[4].size):
        objective += T200[4][group] * x[group]
    return objective, constraints


@njit(cache=True, nogil=True)
def truss18_details(x: np.ndarray) -> tuple[float, np.ndarray]:
    areas = x[:4]
    nodes = np.array([
        [1250.0, 250.0, 0.0], [1000.0, 250.0, 0.0], [x[4], x[5], 0.0],
        [750.0, 250.0, 0.0], [x[6], x[7], 0.0], [500.0, 250.0, 0.0],
        [x[8], x[9], 0.0], [250.0, 250.0, 0.0], [x[10], x[11], 0.0],
        [0.0, 250.0, 0.0], [0.0, 0.0, 0.0],
    ])
    stiffness = np.zeros((T18_FREE_GLOBAL.size, T18_FREE_GLOBAL.size))
    lengths = np.empty(18)
    directions = np.empty((18, 3))
    free_dofs = np.empty((18, 6), dtype=np.int64)
    objective = 0.0
    elastic_modulus = 10_000_000.0
    for e in range(18):
        node_a = T18_ELEMENTS[e, 0]
        node_b = T18_ELEMENTS[e, 1]
        delta = nodes[node_b] - nodes[node_a]
        length = np.sqrt(np.sum(delta * delta))
        direction = delta / length
        lengths[e] = length
        directions[e] = direction
        global_dofs = np.array([
            3 * node_a, 3 * node_a + 1, 3 * node_a + 2,
            3 * node_b, 3 * node_b + 1, 3 * node_b + 2,
        ], dtype=np.int64)
        free_dofs[e] = T18_GLOBAL_TO_FREE[global_dofs]
        vector = np.empty(6)
        for axis in range(3):
            vector[axis] = direction[axis]
            vector[axis + 3] = -direction[axis]
        area = areas[T18_GROUPS[e]]
        objective += length * area * 0.1
        for i in range(6):
            row = free_dofs[e, i]
            if row < 0:
                continue
            for j in range(6):
                col = free_dofs[e, j]
                if col >= 0:
                    stiffness[row, col] += elastic_modulus * area / length * vector[i] * vector[j]

    displacement = np.linalg.solve(stiffness, T18_LOAD)
    stress_sum = 0.0
    buckling_sum = 0.0
    for e in range(18):
        extension = 0.0
        for axis in range(3):
            dof_a = free_dofs[e, axis]
            dof_b = free_dofs[e, axis + 3]
            ua = displacement[dof_a] if dof_a >= 0 else 0.0
            ub = displacement[dof_b] if dof_b >= 0 else 0.0
            extension += directions[e, axis] * (ub - ua)
        stress = elastic_modulus * extension / lengths[e]
        violation = abs(stress) / 20_000.0 - 1.0
        if violation > 0.0:
            stress_sum += violation
        if stress < 0.0:
            buckling_allowable = 4.0 * elastic_modulus * areas[T18_GROUPS[e]] / (lengths[e] ** 2)
            violation = abs(stress) / buckling_allowable - 1.0
            if violation > 0.0:
                buckling_sum += violation
    return objective, np.array([stress_sum, buckling_sum])


@njit(cache=True, nogil=True)
def static_truss_details(problem_id: int, x: np.ndarray) -> tuple[float, np.ndarray]:
    if problem_id == 20:
        return truss120_details(x)
    if problem_id == 21:
        return truss17_details(x)
    if problem_id == 22:
        return truss18_details(x)
    if problem_id == 23:
        return truss25_details(x)
    if problem_id == 24:
        return truss72_details(x)
    if problem_id == 25:
        return truss200_details(x)
    raise ValueError("Unsupported static truss problem id")


@njit(cache=True, nogil=True)
def static_truss_merit(objective: float, constraints: np.ndarray) -> tuple[float, float, float]:
    total = 0.0
    maximum = 0.0
    for value in constraints:
        if value > 0.0:
            total += value
            if value > maximum:
                maximum = value
    return objective + 1.0e8 * total, total, maximum
