"""Python translation of the supplied MATLAB 120 bar space truss model.

The model follows TrussData_120bar.m and Truss_analysis_120bar.m. Lengths are
in inches, areas in square inches, stresses in psi and weight in pounds.
"""

from __future__ import annotations

import numpy as np
from numba import njit


NODES = np.array([
    [0, 0, 275.59],
    [273.26, 0, 196.85], [236.6501, 136.63, 196.85],
    [136.63, 236.6501, 196.85], [0, 273.26, 196.85],
    [-136.63, 236.6501, 196.85], [-236.6501, 136.63, 196.85],
    [-273.26, 0, 196.85], [-236.6501, -136.63, 196.85],
    [-136.63, -236.6501, 196.85], [0, -273.26, 196.85],
    [136.63, -236.6501, 196.85], [236.6501, -136.63, 196.85],
    [492.12, 0, 118.11], [475.3514, 127.37, 118.11],
    [426.1884, 246.06, 118.11], [347.9814, 347.9814, 118.11],
    [246.06, 426.1884, 118.11], [127.37, 475.3514, 118.11],
    [0, 492.12, 118.11], [-127.37, 475.3514, 118.11],
    [-246.06, 426.1884, 118.11], [-347.9814, 347.9814, 118.11],
    [-426.1884, 246.06, 118.11], [-475.3514, 127.37, 118.11],
    [-492.12, 0, 118.11], [-475.3514, -127.37, 118.11],
    [-426.1884, -246.06, 118.11], [-347.9814, -347.9814, 118.11],
    [-246.06, -426.1884, 118.11], [-127.37, -475.3514, 118.11],
    [0, -492.12, 118.11], [127.37, -475.3514, 118.11],
    [246.06, -426.1884, 118.11], [347.9814, -347.9814, 118.11],
    [426.1884, -246.06, 118.11], [475.3514, -127.37, 118.11],
    [625.59, 0, 0], [541.7768, 312.795, 0],
    [312.795, 541.7768, 0], [0, 625.59, 0],
    [-312.795, 541.7768, 0], [-541.7768, 312.795, 0],
    [-625.59, 0, 0], [-541.7768, -312.795, 0],
    [-312.795, -541.7768, 0], [0, -625.59, 0],
    [312.795, -541.7768, 0], [541.7768, -312.795, 0],
], dtype=np.float64)

# MATLAB node numbers converted to zero based Python indices.
ELEMENTS = np.array([
    [1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],[1,10],[1,11],[1,12],[1,13],
    [2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,2],
    [2,14],[3,16],[4,18],[5,20],[6,22],[7,24],[8,26],[9,28],[10,30],[11,32],[12,34],[13,36],
    [2,15],[3,15],[3,17],[4,17],[4,19],[5,19],[5,21],[6,21],[6,23],[7,23],[7,25],[8,25],
    [8,27],[9,27],[9,29],[10,29],[10,31],[11,31],[11,33],[12,33],[12,35],[13,35],[13,37],[2,37],
    [14,15],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[22,23],[23,24],[24,25],
    [25,26],[26,27],[27,28],[28,29],[29,30],[30,31],[31,32],[32,33],[33,34],[34,35],[35,36],[36,37],[37,14],
    [14,38],[16,39],[18,40],[20,41],[22,42],[24,43],[26,44],[28,45],[30,46],[32,47],[34,48],[36,49],
    [15,38],[15,39],[17,39],[17,40],[19,40],[19,41],[21,41],[21,42],[23,42],[23,43],[25,43],[25,44],
    [27,44],[27,45],[29,45],[29,46],[31,46],[31,47],[33,47],[33,48],[35,48],[35,49],[37,49],[37,38],
], dtype=np.int64) - 1

GROUPS = np.empty(120, dtype=np.int64)
GROUPS[0:12] = 0
GROUPS[12:24] = 1
GROUPS[24:36] = 2
GROUPS[36:60] = 3
GROUPS[60:84] = 4
GROUPS[84:96] = 5
GROUPS[96:120] = 6

ELASTIC_MODULUS = 30_450_000.0
YIELD_STRENGTH = 58_000.0
MAX_DISPLACEMENT = 0.1969
UNIT_WEIGHT = 0.288
LOWER_BOUNDS = np.full(7, 0.775, dtype=np.float64)
UPPER_BOUNDS = np.full(7, 20.0, dtype=np.float64)
FREE_DOF_COUNT = 37 * 3


def _precompute() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.empty(120, dtype=np.float64)
    directions = np.empty((120, 3), dtype=np.float64)
    element_dofs = np.empty((120, 6), dtype=np.int64)
    group_stiffness = np.zeros((7, FREE_DOF_COUNT, FREE_DOF_COUNT), dtype=np.float64)
    objective_coefficients = np.zeros(7, dtype=np.float64)

    for e, (node_a, node_b) in enumerate(ELEMENTS):
        delta = NODES[node_b] - NODES[node_a]
        length = float(np.linalg.norm(delta))
        direction = delta / length
        lengths[e] = length
        directions[e] = direction
        dofs = np.array([
            3 * node_a, 3 * node_a + 1, 3 * node_a + 2,
            3 * node_b, 3 * node_b + 1, 3 * node_b + 2,
        ], dtype=np.int64)
        element_dofs[e] = dofs
        vector = np.concatenate((direction, -direction))
        local_global = (ELASTIC_MODULUS / length) * np.outer(vector, vector)
        group = GROUPS[e]
        objective_coefficients[group] += length * UNIT_WEIGHT
        for i in range(6):
            gi = dofs[i]
            if gi >= FREE_DOF_COUNT:
                continue
            for j in range(6):
                gj = dofs[j]
                if gj < FREE_DOF_COUNT:
                    group_stiffness[group, gi, gj] += local_global[i, j]

    load = np.zeros(FREE_DOF_COUNT, dtype=np.float64)
    load[2] = -13_490.0
    for node in range(1, 14):
        load[3 * node + 2] = -6_744.0
    for node in range(14, 37):
        load[3 * node + 2] = -2_248.0
    return lengths, directions, element_dofs, group_stiffness, objective_coefficients, load


LENGTHS, DIRECTIONS, ELEMENT_DOFS, GROUP_STIFFNESS, OBJECTIVE_COEFFICIENTS, LOAD = _precompute()


@njit(cache=True, nogil=True)
def truss120_details(areas: np.ndarray) -> tuple[float, np.ndarray]:
    """Return structural weight and the two aggregated MATLAB violations."""
    stiffness = np.zeros((FREE_DOF_COUNT, FREE_DOF_COUNT), dtype=np.float64)
    for group in range(7):
        stiffness += areas[group] * GROUP_STIFFNESS[group]
    free_displacement = np.linalg.solve(stiffness, LOAD)

    stress_violation = 0.0
    tension_allowable = 0.6 * YIELD_STRENGTH
    alpha = 0.4993
    beta = 0.6777
    cc = np.sqrt(2.0 * np.pi * np.pi * ELASTIC_MODULUS / YIELD_STRENGTH)
    for e in range(120):
        dofs = ELEMENT_DOFS[e]
        extension = 0.0
        for axis in range(3):
            ua = free_displacement[dofs[axis]] if dofs[axis] < FREE_DOF_COUNT else 0.0
            ub = free_displacement[dofs[axis + 3]] if dofs[axis + 3] < FREE_DOF_COUNT else 0.0
            extension += DIRECTIONS[e, axis] * (ub - ua)
        stress = ELASTIC_MODULUS * extension / LENGTHS[e]
        if stress < 0.0:
            area = areas[GROUPS[e]]
            radius = alpha * area ** beta
            slenderness = LENGTHS[e] / radius
            if slenderness < cc:
                ratio = slenderness / cc
                allowable = (
                    (1.0 - 0.5 * ratio * ratio) * YIELD_STRENGTH
                    / (5.0 / 3.0 + 3.0 * slenderness / (8.0 * cc)
                       - slenderness ** 3 / (8.0 * cc ** 3))
                )
            else:
                allowable = 12.0 * np.pi * np.pi * ELASTIC_MODULUS / (23.0 * slenderness * slenderness)
        else:
            allowable = tension_allowable
        violation = abs(stress) / allowable - 1.0
        if violation > 0.0:
            stress_violation += violation

    displacement_violation = 0.0
    for value in free_displacement:
        violation = abs(value) / MAX_DISPLACEMENT - 1.0
        if violation > 0.0:
            displacement_violation += violation

    objective = 0.0
    for group in range(7):
        objective += OBJECTIVE_COEFFICIENTS[group] * areas[group]
    return objective, np.array([stress_violation, displacement_violation])


@njit(cache=True, nogil=True)
def truss120_matlab_merit(objective: float, constraints: np.ndarray) -> tuple[float, float, float]:
    """Reproduce the 1e8 additive penalty in Truss_analysis_120bar.m."""
    total = 0.0
    maximum = 0.0
    for value in constraints:
        if value > 0.0:
            total += value
            if value > maximum:
                maximum = value
    return objective + 1.0e8 * total, total, maximum

