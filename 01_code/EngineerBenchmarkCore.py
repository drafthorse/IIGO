"""Constrained engineering benchmark models.

Optimizers search in ``[0, 1] ** D``.  This module maps normalized candidates
to physical variables, applies deterministic discrete-variable handling, and
evaluates the original objective and constraints.  Every constraint follows
the convention ``g(x) <= 0``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple, Union

import numpy as np
from numba import njit

from StaticTrussBenchmarks import static_truss_details, static_truss_merit


ENGINEERING_BENCHMARK_ID = 9000
INFEASIBLE_MERIT_BASE = 1.0e10
FEASIBILITY_TOL = 1.0e-8

_CANONICAL_NAMES: Dict[int, str] = {
    2: "WBD",
    3: "SRD",
    6: "CBD",
    17: "RSLD",
    18: "ELD",
    19: "RSLD_COMPLEX",
    20: "TRUSS120",
    21: "TRUSS17",
    22: "TRUSS18",
    23: "TRUSS25",
    24: "TRUSS72",
    25: "TRUSS200",
    26: "RSLD_LEGACY",
}
_NAME_TO_ID: Dict[str, int] = {
    "WBD": 2,
    "WELDED_BEAM": 2,
    "WELDED_BEAM_DESIGN": 2,
    "CBD": 6,
    "CANTILEVER_BEAM": 6,
    "CANTILEVER_BEAM_DESIGN": 6,
    "SRD": 3,
    "SPEED_REDUCER": 3,
    "SPEED_REDUCER_DESIGN": 3,
    "RSD": 17,
    "RSLD": 17,
    "ROTATING_SHAFT": 17,
    "ROTATING_SHAFT_DESIGN": 17,
    "ROTATING_SHAFT_LIGHTWEIGHT_DESIGN": 17,
    "SHAFT": 17,
    "RSLD_LEGACY": 26,
    "LEGACY_RSLD": 26,
    "ELD": 18,
    "ECONOMIC_LOAD_DISPATCH": 18,
    "SIX_UNIT_ELD": 18,
    "6_UNIT_ELD": 18,
    "SIX_UNIT_VALVE_POINT_ELD": 18,
    "RSLD_COMPLEX": 19,
    "RSLD_V2": 19,
    "SINGLE_POINT_TEST_RIG": 19,
    "SINGLE_POINT_TEST_RIG_LIGHTWEIGHT_DESIGN": 19,
    "TRUSS120": 20,
    "120_BAR_TRUSS": 20,
    "120_BAR_SPACE_TRUSS": 20,
    "TRUSS17": 21,
    "17_BAR_TRUSS": 21,
    "TRUSS18": 22,
    "18_BAR_TRUSS": 22,
    "TRUSS25": 23,
    "25_BAR_TRUSS": 23,
    "TRUSS72": 24,
    "72_BAR_TRUSS": 24,
    "TRUSS200": 25,
    "200_BAR_TRUSS": 25,
}


def benchmark_to_number(name: Union[str, int]) -> int:
    if isinstance(name, (int, np.integer)):
        problem_id = int(name)
        if problem_id not in _CANONICAL_NAMES:
            raise ValueError(f"Unsupported engineering problem id: {problem_id}")
        return problem_id
    key = str(name).strip().upper().replace("-", "_").replace(" ", "_")
    if key not in _NAME_TO_ID:
        supported = ", ".join(_CANONICAL_NAMES.values())
        raise ValueError(f"Unsupported engineering problem {name!r}; use {supported}")
    return _NAME_TO_ID[key]


# Backward-compatible function alias.
benchamrks_to_number = benchmark_to_number


def problem_name(problem: Union[str, int]) -> str:
    return _CANONICAL_NAMES[benchmark_to_number(problem)]


def list_engineering_problems() -> Tuple[str, ...]:
    return (
        "WBD", "CBD", "SRD", "RSLD", "ELD",
        "TRUSS17", "TRUSS18", "TRUSS25", "TRUSS72", "TRUSS120", "TRUSS200",
    )


def get_eld_parameters() -> Dict[str, object]:
    """Return the 6-unit ELD parameters used by the evaluator."""
    return {
        "quadratic": np.array([0.0070, 0.0095, 0.0090, 0.0090, 0.0080, 0.0075]),
        "linear": np.array([7.0, 10.0, 8.5, 11.0, 10.5, 12.0]),
        "constant": np.array([240.0, 200.0, 220.0, 200.0, 220.0, 190.0]),
        "valve_amplitude": np.array([300.0, 200.0, 200.0, 150.0, 150.0, 150.0]),
        "valve_frequency": np.array([0.035, 0.042, 0.042, 0.063, 0.063, 0.063]),
        "minimum_mw": np.array([100.0, 50.0, 80.0, 50.0, 50.0, 50.0]),
        "maximum_mw": np.array([500.0, 200.0, 300.0, 150.0, 200.0, 120.0]),
        "demand_mw": 1263.0,
        "balance_tolerance_mw": 0.01,
        "transmission_losses_included": False,
    }


def get_rsld_parameters() -> Dict[str, object]:
    """Return the single-point test-rig shaft design parameters.

    The load case is derived from calculation document B260210000-CAL-001
    Rev. B.  The printed bending-moment value in that document is inconsistent
    with its own equation, so the value below is recomputed as W * L1 / 2.
    """
    vertical_load_n = 221.0e3
    bending_arm_mm = 175.0
    torsion_arm_mm = 250.0
    return {
        "density_kg_m3": 7850.0,
        "young_modulus_mpa": 206000.0,
        "poisson_ratio": 0.3,
        "yield_strength_mpa": 835.0,
        "allowable_shear_mpa": 334.0,
        "allowable_equivalent_stress_mpa": 627.0,
        "vertical_load_n": vertical_load_n,
        "bending_arm_mm": bending_arm_mm,
        "torsion_arm_mm": torsion_arm_mm,
        "bending_moment_n_mm": vertical_load_n * bending_arm_mm * 0.5,
        "torque_n_mm": vertical_load_n * torsion_arm_mm,
        "minimum_wall_thickness_mm": 20.0,
        "baseline_outer_diameter_mm": 200.0,
        "baseline_inner_to_outer_ratio": 0.0,
    }


def get_rsld_complex_parameters() -> Dict[str, object]:
    """Return the reduced-order data for the coupled test-rig model.

    Baseline masses, finite-element response envelopes, rotational inertia,
    bolt data, and shaft loads are taken from B260210000-CAL-001 Rev. B.
    The three dimensionless section scales are reduced-order design variables;
    they are not replacements for a final finite-element verification.
    """
    return {
        "payload_n": 140000.0,
        "gravity_n_per_kg": 10.0,
        "lifting_mass_kg": 2600.0,
        "swing_mass_kg": 5500.0,
        "base_mass_kg": 9000.0,
        "counterweight_mass_kg": 16600.0,
        "allowable_structural_stress_mpa": 266.9,
        "lifting_stress_envelope_mpa": 164.0,
        "swing_stress_envelope_mpa": 109.8,
        "base_stress_envelope_mpa": 82.2,
        "lifting_deformation_envelope_mm": 2.5086,
        "swing_deformation_envelope_mm": 3.0955,
        "base_deformation_envelope_mm": 1.9624,
        "deformation_allowance_factor": 1.2,
        "shaft_reference_length_m": 1.0,
        "shaft_density_kg_m3": 7850.0,
        "shaft_allowable_shear_mpa": 334.0,
        "shaft_allowable_equivalent_stress_mpa": 627.0,
        "shaft_bending_arm_mm": 175.0,
        "shaft_torsion_arm_mm": 250.0,
        "shaft_minimum_wall_thickness_mm": 20.0,
        "baseline_rotational_inertia_kg_m2": 72105.0,
        "lifting_inertia_radius_m": 0.33,
        "swing_inertia_radius_m": 1.275,
        "angular_acceleration_rad_s2": 1.745,
        "anchor_bolt_count": 8,
        "anchor_bolt_radius_m": 1.275,
        "anchor_bolt_capacity_n": 212000.0,
    }


@njit(cache=True)
def _get_bounds(problem_id: int) -> Tuple[np.ndarray, np.ndarray]:
    if problem_id == 2:  # h, l, t, b
        lower = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)
        upper = np.array([2.0, 10.0, 10.0, 2.0], dtype=np.float64)
    elif problem_id == 3:  # gear width, module, teeth, lengths and diameters
        lower = np.array([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0])
        upper = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])
    elif problem_id == 6:  # five segment side lengths
        lower = np.full(5, 0.01, dtype=np.float64)
        upper = np.full(5, 100.0, dtype=np.float64)
    elif problem_id == 17:  # outer diameter (mm), inner/outer diameter ratio
        lower = np.array([80.0, 0.0])
        upper = np.array([200.0, 0.8])
    elif problem_id == 18:  # six generating-unit outputs (MW)
        lower = np.array([100.0, 50.0, 80.0, 50.0, 50.0, 50.0])
        upper = np.array([500.0, 200.0, 300.0, 150.0, 200.0, 120.0])
    elif problem_id == 19:  # three section scales, shaft D (mm), bore ratio
        lower = np.array([0.70, 0.65, 0.60, 80.0, 0.0])
        upper = np.array([1.10, 1.10, 1.10, 200.0, 0.8])
    elif problem_id == 20:  # seven cross-sectional groups of the 120-bar truss
        lower = np.full(7, 0.775, dtype=np.float64)
        upper = np.full(7, 20.0, dtype=np.float64)
    elif problem_id == 21:  # 17 independent areas
        lower = np.full(17, 0.1, dtype=np.float64)
        upper = np.full(17, 50.0, dtype=np.float64)
    elif problem_id == 22:  # four areas and eight free node coordinates
        lower = np.array([3.5, 3.5, 3.5, 3.5, 775.0, -225.0, 525.0, -225.0,
                          275.0, -225.0, 25.0, -225.0])
        upper = np.array([18.0, 18.0, 18.0, 18.0, 1225.0, 245.0, 975.0, 245.0,
                          725.0, 245.0, 475.0, 245.0])
    elif problem_id == 23:  # eight area groups
        lower = np.full(8, 0.01, dtype=np.float64)
        upper = np.full(8, 3.4, dtype=np.float64)
    elif problem_id == 24:  # sixteen area groups
        lower = np.full(16, 0.1, dtype=np.float64)
        upper = np.full(16, 20.0, dtype=np.float64)
    elif problem_id == 25:  # twenty-nine area groups
        lower = np.full(29, 0.1, dtype=np.float64)
        upper = np.full(29, 20.0, dtype=np.float64)
    elif problem_id == 26:  # shaft diameter and lengths: d, L1 and L3
        lower = np.array([100.0, 150.0, 200.0], dtype=np.float64)
        upper = np.array([120.0, 200.0, 250.0], dtype=np.float64)
    else:
        raise ValueError("Unsupported engineering problem id")
    return lower, upper


def get_bounds(problem: Union[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    lower, upper = _get_bounds(benchmark_to_number(problem))
    return lower.copy(), upper.copy()


def get_boundary(problem: Union[str, int]) -> np.ndarray:
    """Legacy boundary matrix in the original ``[upper; lower]`` order."""
    lower, upper = get_bounds(problem)
    return np.vstack((upper, lower))


def problem_dimension(problem: Union[str, int]) -> int:
    lower, _ = get_bounds(problem)
    return int(lower.size)


@njit(cache=True)
def _project_box_sum(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_sum: float,
) -> np.ndarray:
    """Euclidean projection onto box bounds with an exact prescribed sum."""
    if target_sum < np.sum(lower) or target_sum > np.sum(upper):
        raise ValueError("Requested ELD load is outside aggregate generation limits")

    lambda_low = values[0] - upper[0]
    lambda_high = values[0] - lower[0]
    for j in range(1, values.size):
        lambda_low = min(lambda_low, values[j] - upper[j])
        lambda_high = max(lambda_high, values[j] - lower[j])

    projected = np.empty(values.size, dtype=np.float64)
    for _ in range(100):
        multiplier = 0.5 * (lambda_low + lambda_high)
        current_sum = 0.0
        for j in range(values.size):
            value = values[j] - multiplier
            if value < lower[j]:
                value = lower[j]
            elif value > upper[j]:
                value = upper[j]
            projected[j] = value
            current_sum += value
        if current_sum > target_sum:
            lambda_low = multiplier
        else:
            lambda_high = multiplier

    # Remove the last few floating-point ulps without violating any bound.
    residual = target_sum - np.sum(projected)
    if residual > 0.0:
        for j in range(projected.size):
            room = upper[j] - projected[j]
            change = min(room, residual)
            projected[j] += change
            residual -= change
            if residual <= 1.0e-12:
                break
    elif residual < 0.0:
        residual = -residual
        for j in range(projected.size):
            room = projected[j] - lower[j]
            change = min(room, residual)
            projected[j] -= change
            residual -= change
            if residual <= 1.0e-12:
                break
    return projected


@njit(cache=True)
def decode_candidate(problem_id: int, encoded: np.ndarray) -> np.ndarray:
    lower, upper = _get_bounds(problem_id)
    if encoded.size != lower.size:
        raise ValueError("Candidate dimension does not match engineering problem")
    physical = np.empty(lower.size, dtype=np.float64)
    for j in range(lower.size):
        value = encoded[j]
        if value < 0.0:
            value = 0.0
        elif value > 1.0:
            value = 1.0
        physical[j] = lower[j] + value * (upper[j] - lower[j])
    if problem_id == 3:
        physical[2] = np.floor(physical[2] + 0.5)
    elif problem_id == 18:
        physical = _project_box_sum(physical, lower, upper, 1263.0)
    return physical


@njit(cache=True)
def encode_candidate(problem_id: int, physical: np.ndarray) -> np.ndarray:
    lower, upper = _get_bounds(problem_id)
    if physical.size != lower.size:
        raise ValueError("Candidate dimension does not match engineering problem")
    encoded = (physical - lower) / (upper - lower)
    for j in range(encoded.size):
        if encoded[j] < 0.0:
            encoded[j] = 0.0
        elif encoded[j] > 1.0:
            encoded[j] = 1.0
    return encoded


@njit(cache=True)
def _wbd_details(x: np.ndarray) -> Tuple[float, np.ndarray]:
    h, length, t, b = x
    p = 6000.0
    beam_length = 14.0
    young = 30.0e6
    shear_modulus = 12.0e6
    tau_limit = 13600.0
    sigma_limit = 30000.0
    delta_limit = 0.25

    objective = 1.10471 * h * h * length + 0.04811 * t * b * (14.0 + length)
    tau_prime = p / (np.sqrt(2.0) * h * length)
    moment = p * (beam_length + length / 2.0)
    radius = np.sqrt(length * length / 4.0 + ((h + t) / 2.0) ** 2)
    polar = 2.0 * np.sqrt(2.0) * h * length * (
        length * length / 12.0 + ((h + t) / 2.0) ** 2
    )
    tau_second = moment * radius / polar
    tau = np.sqrt(
        tau_prime**2
        + 2.0 * tau_prime * tau_second * length / (2.0 * radius)
        + tau_second**2
    )
    sigma = 6.0 * p * beam_length / (b * t**2)
    delta = 4.0 * p * beam_length**3 / (young * t**3 * b)
    critical_load = (
        4.013
        * young
        * np.sqrt(t**2 * b**6 / 36.0)
        / beam_length**2
        * (1.0 - t / (2.0 * beam_length) * np.sqrt(young / (4.0 * shear_modulus)))
    )
    constraints = np.array(
        [
            tau / tau_limit - 1.0,
            sigma / sigma_limit - 1.0,
            delta / delta_limit - 1.0,
            (h - b) / 2.0,
            p / critical_load - 1.0,
            (0.125 - h) / 0.125,
            (0.10471 * h**2 + 0.04811 * t * b * (14.0 + length)) / 5.0 - 1.0,
        ]
    )
    return objective, constraints


@njit(cache=True)
def _cbd_details(x: np.ndarray) -> Tuple[float, np.ndarray]:
    objective = 0.0624 * np.sum(x)
    constraint = (
        61.0 / x[0] ** 3
        + 37.0 / x[1] ** 3
        + 19.0 / x[2] ** 3
        + 7.0 / x[3] ** 3
        + 1.0 / x[4] ** 3
        - 1.0
    )
    return objective, np.array([constraint])


@njit(cache=True)
def _srd_details(x: np.ndarray) -> Tuple[float, np.ndarray]:
    z1, z2, z3, z4, z5, z6, z7 = x
    objective = (
        0.7854 * z1 * z2**2 * (3.3333 * z3**2 + 14.9334 * z3 - 43.0934)
        - 1.508 * z1 * (z6**2 + z7**2)
        + 7.4777 * (z6**3 + z7**3)
        + 0.7854 * (z4 * z6**2 + z5 * z7**2)
    )
    constraints = np.array(
        [
            27.0 / (z1 * z2**2 * z3) - 1.0,
            397.5 / (z1 * z2**2 * z3**2) - 1.0,
            1.93 * z4**3 / (z2 * z6**4 * z3) - 1.0,
            1.93 * z5**3 / (z2 * z7**4 * z3) - 1.0,
            np.sqrt((745.0 * z4 / (z2 * z3)) ** 2 + 16.9e6) / (110.0 * z6**3) - 1.0,
            np.sqrt((745.0 * z5 / (z2 * z3)) ** 2 + 157.5e6) / (85.0 * z7**3) - 1.0,
            z2 * z3 / 40.0 - 1.0,
            5.0 * z2 / z1 - 1.0,
            z1 / (12.0 * z2) - 1.0,
            (1.5 * z6 + 1.9) / z4 - 1.0,
            (1.1 * z7 + 1.9) / z5 - 1.0,
        ]
    )
    return objective, constraints


@njit(cache=True)
def _rsd_details(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """Lightweight pivot-shaft design for the single-point test apparatus.

    Variables are the shaft outer diameter D in mm and the bore ratio k=d_i/D.
    The objective is shaft mass per unit length in kg/m.  Loads, material data,
    and lever arms follow B260210000-CAL-001 Rev. B.  The equivalent-stress
    check uses the von Mises form rather than the document's unweighted
    root-sum-square expression.
    """
    outer_diameter, diameter_ratio = x
    inner_diameter = diameter_ratio * outer_diameter
    density = 7850.0
    vertical_load = 221.0e3
    bending_arm = 175.0
    torsion_arm = 250.0
    torque = vertical_load * torsion_arm
    bending_moment = vertical_load * bending_arm * 0.5
    tau_limit = 334.0
    sigma_limit = 627.0
    minimum_wall = 20.0

    area = 0.25 * np.pi * (outer_diameter**2 - inner_diameter**2)
    torsional_modulus = (
        np.pi
        * (outer_diameter**4 - inner_diameter**4)
        / (16.0 * outer_diameter)
    )
    bending_modulus = 0.5 * torsional_modulus

    objective = density * area * 1.0e-6
    transverse_shear = vertical_load / area
    torsional_shear = torque / torsional_modulus
    combined_shear = np.sqrt(transverse_shear**2 + torsional_shear**2)
    bending_stress = bending_moment / bending_modulus
    equivalent_stress = np.sqrt(bending_stress**2 + 3.0 * combined_shear**2)
    wall_thickness = 0.5 * (outer_diameter - inner_diameter)
    constraints = np.array(
        [
            combined_shear / tau_limit - 1.0,
            equivalent_stress / sigma_limit - 1.0,
            minimum_wall / wall_thickness - 1.0,
        ]
    )
    return objective, constraints


@njit(cache=True)
def _rsld_legacy_details(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """Three-variable rotating-shaft design model."""
    diameter, length_1, length_3 = x
    density = 7850.0
    vertical_load = 221.0e3
    torque = 55.250e6
    bending_moment = 82.870e6
    young_modulus = 206000.0

    objective = (
        density * np.pi * diameter**2 * (length_1 + length_3) * 0.25e-9
    )
    transverse_shear = 4.0 * vertical_load / (np.pi * diameter**2)
    torsional_shear = 16.0 * torque / (np.pi * diameter**3)
    bending_stress = 32.0 * bending_moment / (np.pi * diameter**3)
    shear_stress = np.sqrt(transverse_shear**2 + torsional_shear**2)
    equivalent_stress = np.sqrt(
        bending_stress**2 + transverse_shear**2 + torsional_shear**2
    )
    deflection = (
        64.0 * vertical_load * length_1**3
        / (3.0 * young_modulus * np.pi * diameter**4)
    )
    constraints = np.array(
        [
            shear_stress / 334.0 - 1.0,
            equivalent_stress / 627.0 - 1.0,
            deflection / 3.0 - 1.0,
        ]
    )
    return objective, constraints


@njit(cache=True)
def _rsld_complex_details(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """Reduced coupled design of the single-point test apparatus.

    Variables are the section scales of the lifting platform, swing platform,
    and base, followed by pivot-shaft outer diameter and bore ratio.  Structural
    mass scales with cross-sectional area, stress with section modulus, and
    displacement with second moment of area.  The finite-element values in the
    source calculation are used as baseline response envelopes.  The governing
    responses are represented by three normalized envelope constraints.
    """
    lift_scale, swing_scale, base_scale, outer_diameter, diameter_ratio = x

    payload = 140000.0
    gravity = 10.0
    lift_mass_0 = 2600.0
    swing_mass_0 = 5500.0
    base_mass_0 = 9000.0
    counterweight_mass = 16600.0

    lift_mass = lift_mass_0 * lift_scale**2
    swing_mass = swing_mass_0 * swing_scale**2
    base_mass = base_mass_0 * base_scale**2

    lift_load = payload + gravity * lift_mass
    swing_load = payload + gravity * (lift_mass + swing_mass)
    base_load = (
        payload
        + gravity * (lift_mass + swing_mass + base_mass + counterweight_mass)
    )
    lift_load_0 = payload + gravity * lift_mass_0
    swing_load_0 = payload + gravity * (lift_mass_0 + swing_mass_0)
    base_load_0 = (
        payload
        + gravity
        * (lift_mass_0 + swing_mass_0 + base_mass_0 + counterweight_mass)
    )

    lift_stress = 164.0 * (lift_load / lift_load_0) / lift_scale**3
    swing_stress = 109.8 * (swing_load / swing_load_0) / swing_scale**3
    base_stress = 82.2 * (base_load / base_load_0) / base_scale**3

    lift_deformation_0 = 2.5086
    swing_deformation_0 = 3.0955
    base_deformation_0 = 1.9624
    deformation_factor = 1.2
    lift_deformation = (
        lift_deformation_0 * (lift_load / lift_load_0) / lift_scale**4
    )
    swing_deformation = (
        swing_deformation_0 * (swing_load / swing_load_0) / swing_scale**4
    )
    base_deformation = (
        base_deformation_0 * (base_load / base_load_0) / base_scale**4
    )

    inner_diameter = diameter_ratio * outer_diameter
    shaft_area = 0.25 * np.pi * (outer_diameter**2 - inner_diameter**2)
    torsional_modulus = (
        np.pi
        * (outer_diameter**4 - inner_diameter**4)
        / (16.0 * outer_diameter)
    )
    bending_modulus = 0.5 * torsional_modulus
    shaft_load = swing_load
    shaft_torque = shaft_load * 250.0
    shaft_bending_moment = shaft_load * 175.0 * 0.5
    direct_shear = shaft_load / shaft_area
    torsional_shear = shaft_torque / torsional_modulus
    combined_shear = np.sqrt(direct_shear**2 + torsional_shear**2)
    bending_stress = shaft_bending_moment / bending_modulus
    equivalent_stress = np.sqrt(bending_stress**2 + 3.0 * combined_shear**2)
    wall_thickness = 0.5 * (outer_diameter - inner_diameter)
    shaft_mass = 7850.0 * shaft_area * 1.0e-6

    # Preserve the source document's baseline inertia while coupling changes in
    # the two optimized moving masses to the dynamic bolt load.
    rotational_inertia = (
        72105.0
        + (lift_mass - lift_mass_0) * 0.33**2
        + (swing_mass - swing_mass_0) * 1.275**2
    )
    inertial_torque = rotational_inertia * 1.745
    anchor_bolt_tension = inertial_torque / (8.0 * 1.275)

    objective = lift_mass + swing_mass + base_mass + shaft_mass
    deformation_envelope = max(
        lift_deformation / (deformation_factor * lift_deformation_0),
        swing_deformation / (deformation_factor * swing_deformation_0),
        base_deformation / (deformation_factor * base_deformation_0),
    )
    shaft_strength_envelope = max(
        combined_shear / 334.0,
        equivalent_stress / 627.0,
    )
    constraints = np.array(
        [
            deformation_envelope - 1.0,
            shaft_strength_envelope - 1.0,
            20.0 / wall_thickness - 1.0,
        ]
    )
    return objective, constraints


@njit(cache=True)
def _eld_details(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """Classic 6-unit valve-point ELD without transmission losses."""
    quadratic = np.array([0.0070, 0.0095, 0.0090, 0.0090, 0.0080, 0.0075])
    linear = np.array([7.0, 10.0, 8.5, 11.0, 10.5, 12.0])
    constant = np.array([240.0, 200.0, 220.0, 200.0, 220.0, 190.0])
    valve_amplitude = np.array([300.0, 200.0, 200.0, 150.0, 150.0, 150.0])
    valve_frequency = np.array([0.035, 0.042, 0.042, 0.063, 0.063, 0.063])
    minimum_output = np.array([100.0, 50.0, 80.0, 50.0, 50.0, 50.0])

    objective = 0.0
    for i in range(6):
        objective += (
            quadratic[i] * x[i] ** 2
            + linear[i] * x[i]
            + constant[i]
            + abs(
                valve_amplitude[i]
                * np.sin(valve_frequency[i] * (minimum_output[i] - x[i]))
            )
        )

    # Power-balance tolerance is 0.01 MW. Bounds are enforced by the
    # normalized decoder and checked before this function is called.
    balance_residual = abs(np.sum(x) - 1263.0)
    constraints = np.array([balance_residual / 0.01 - 1.0])
    return objective, constraints


@njit(cache=True)
def evaluate_physical(problem_id: int, physical: np.ndarray) -> Tuple[float, np.ndarray]:
    lower, upper = _get_bounds(problem_id)
    if physical.size != lower.size:
        raise ValueError("Candidate dimension does not match engineering problem")
    x = physical.copy()
    for j in range(x.size):
        if x[j] < lower[j] - 1.0e-12 or x[j] > upper[j] + 1.0e-12:
            raise ValueError("Physical candidate is outside the problem bounds")
        if x[j] < lower[j]:
            x[j] = lower[j]
        elif x[j] > upper[j]:
            x[j] = upper[j]
    if problem_id == 3:
        x[2] = np.floor(x[2] + 0.5)
    if problem_id == 2:
        return _wbd_details(x)
    if problem_id == 3:
        return _srd_details(x)
    if problem_id == 6:
        return _cbd_details(x)
    if problem_id == 17:
        return _rsd_details(x)
    if problem_id == 18:
        return _eld_details(x)
    if problem_id == 19:
        return _rsld_complex_details(x)
    if problem_id == 26:
        return _rsld_legacy_details(x)
    if 20 <= problem_id <= 25:
        return static_truss_details(problem_id, x)
    raise ValueError("Unsupported engineering problem id")


@njit(cache=True)
def constraint_violation(constraints: np.ndarray) -> Tuple[float, float]:
    total = 0.0
    maximum = 0.0
    for value in constraints:
        if value > 0.0:
            total += value * value
            if value > maximum:
                maximum = value
    return total, maximum


@njit(cache=True)
def engineering_merit(objective: float, constraints: np.ndarray) -> Tuple[float, float, float]:
    total, maximum = constraint_violation(constraints)
    if maximum <= FEASIBILITY_TOL:
        return objective, total, maximum
    merit = INFEASIBLE_MERIT_BASE * (1.0 + total + maximum)
    return merit, total, maximum


@njit(cache=True)
def evaluate_encoded(problem_id: int, encoded: np.ndarray) -> Tuple[float, float, float, float]:
    physical = decode_candidate(problem_id, encoded)
    objective, constraints = evaluate_physical(problem_id, physical)
    if 20 <= problem_id <= 25:
        merit, total, maximum = static_truss_merit(objective, constraints)
    else:
        merit, total, maximum = engineering_merit(objective, constraints)
    return merit, objective, total, maximum


@njit(cache=True)
def evaluate_population_encoded(population: np.ndarray, problem_id: int) -> np.ndarray:
    result = np.empty((population.shape[0], 2), dtype=np.float64)
    for i in range(population.shape[0]):
        merit, _, _, _ = evaluate_encoded(problem_id, population[i])
        result[i, 0] = merit
        result[i, 1] = i
    return result


def run_Engineering(problem: Union[str, int], physical: Iterable[float]) -> Tuple[float, float]:
    problem_id = benchmark_to_number(problem)
    x = np.asarray(tuple(physical), dtype=np.float64)
    objective, constraints = evaluate_physical(problem_id, x)
    if 20 <= problem_id <= 25:
        merit, total, _ = static_truss_merit(objective, constraints)
    else:
        merit, total, _ = engineering_merit(objective, constraints)
    return float(merit), float(total)


def evaluate_solution(
    problem: Union[str, int], candidate: Iterable[float], *, encoded: bool = False
) -> Dict[str, object]:
    problem_id = benchmark_to_number(problem)
    values = np.asarray(tuple(candidate), dtype=np.float64)
    physical = decode_candidate(problem_id, values) if encoded else values.copy()
    if problem_id == 3:
        physical[2] = np.floor(physical[2] + 0.5)
    objective, constraints = evaluate_physical(problem_id, physical)
    if 20 <= problem_id <= 25:
        merit, total, maximum = static_truss_merit(objective, constraints)
    else:
        merit, total, maximum = engineering_merit(objective, constraints)
    result = {
        "problem": problem_name(problem_id),
        "problem_id": problem_id,
        "physical_solution": physical,
        "objective": float(objective),
        "constraints": np.asarray(constraints),
        "total_violation": float(total),
        "max_violation": float(maximum),
        "feasible": bool(maximum <= FEASIBILITY_TOL),
        "merit": float(merit),
    }
    if problem_id == 18:
        result.update(
            {
                "test_system": "classic 6-unit valve-point ELD without transmission losses",
                "demand_mw": 1263.0,
                "power_balance_residual_mw": float(np.sum(physical) - 1263.0),
                "balance_tolerance_mw": 0.01,
                "reported_cost_includes_penalty": False,
            }
        )
    elif problem_id == 17:
        outer_diameter, diameter_ratio = physical
        result.update(
            {
                "application": "single-point test apparatus pivot shaft",
                "objective_unit": "kg/m",
                "outer_diameter_mm": float(outer_diameter),
                "inner_diameter_mm": float(outer_diameter * diameter_ratio),
                "inner_to_outer_ratio": float(diameter_ratio),
                "wall_thickness_mm": float(
                    0.5 * outer_diameter * (1.0 - diameter_ratio)
                ),
                "combined_shear_mpa": float(334.0 * (constraints[0] + 1.0)),
                "equivalent_stress_mpa": float(627.0 * (constraints[1] + 1.0)),
            }
        )
    elif problem_id == 19:
        lift_scale, swing_scale, base_scale, outer_diameter, diameter_ratio = physical
        lift_mass = 2600.0 * lift_scale**2
        swing_mass = 5500.0 * swing_scale**2
        base_mass = 9000.0 * base_scale**2
        payload = 140000.0
        gravity = 10.0
        lift_load = payload + gravity * lift_mass
        swing_load = payload + gravity * (lift_mass + swing_mass)
        base_load = payload + gravity * (
            lift_mass + swing_mass + base_mass + 16600.0
        )
        lift_load_0 = 166000.0
        swing_load_0 = 221000.0
        base_load_0 = 477000.0
        inner_diameter = outer_diameter * diameter_ratio
        area = 0.25 * np.pi * (outer_diameter**2 - inner_diameter**2)
        torsional_modulus = np.pi * (
            outer_diameter**4 - inner_diameter**4
        ) / (16.0 * outer_diameter)
        bending_modulus = 0.5 * torsional_modulus
        direct_shear = swing_load / area
        torsional_shear = swing_load * 250.0 / torsional_modulus
        combined_shear = np.sqrt(direct_shear**2 + torsional_shear**2)
        equivalent_stress = np.sqrt(
            (swing_load * 175.0 * 0.5 / bending_modulus) ** 2
            + 3.0 * combined_shear**2
        )
        rotational_inertia = (
            72105.0
            + (lift_mass - 2600.0) * 0.33**2
            + (swing_mass - 5500.0) * 1.275**2
        )
        result.update(
            {
                "application": "coupled lightweight design of a single-point test apparatus",
                "objective_unit": "kg equivalent structural mass",
                "lifting_section_scale": float(lift_scale),
                "swing_section_scale": float(swing_scale),
                "base_section_scale": float(base_scale),
                "lifting_mass_kg": float(lift_mass),
                "swing_mass_kg": float(swing_mass),
                "base_mass_kg": float(base_mass),
                "shaft_mass_for_1m_kg": float(7850.0 * area * 1.0e-6),
                "outer_diameter_mm": float(outer_diameter),
                "inner_diameter_mm": float(inner_diameter),
                "wall_thickness_mm": float(0.5 * (outer_diameter - inner_diameter)),
                "lifting_stress_mpa": float(164.0 * (lift_load / lift_load_0) / lift_scale**3),
                "swing_stress_mpa": float(109.8 * (swing_load / swing_load_0) / swing_scale**3),
                "base_stress_mpa": float(82.2 * (base_load / base_load_0) / base_scale**3),
                "lifting_deformation_mm": float(2.5086 * (lift_load / lift_load_0) / lift_scale**4),
                "swing_deformation_mm": float(3.0955 * (swing_load / swing_load_0) / swing_scale**4),
                "base_deformation_mm": float(1.9624 * (base_load / base_load_0) / base_scale**4),
                "shaft_combined_shear_mpa": float(combined_shear),
                "shaft_equivalent_stress_mpa": float(equivalent_stress),
                "rotational_inertia_kg_m2": float(rotational_inertia),
                "anchor_bolt_tension_n": float(rotational_inertia * 1.745 / (8.0 * 1.275)),
            }
        )
    elif 20 <= problem_id <= 25:
        result.update(
            {
                "application": f"{problem_name(problem_id).replace('TRUSS', '')}-bar truss weight minimization",
                "objective_unit": "lb",
                "cross_sectional_areas_in2": physical.copy(),
                "aggregated_constraint_violations": np.asarray(constraints).copy(),
                "penalty_definition": "objective + 1e8 * total positive violation",
            }
        )
    return result


def WBD(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("WBD", x)


def CBD(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("CBD", x)


def SRD(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("SRD", x)


def RSD(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("RSD", x)


def RSLD(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("RSLD", x)


def ELD(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("ELD", x)


def RSLD_COMPLEX(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("RSLD_COMPLEX", x)


def TRUSS120(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("TRUSS120", x)


def TRUSS17(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("TRUSS17", x)


def TRUSS18(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("TRUSS18", x)


def TRUSS25(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("TRUSS25", x)


def TRUSS72(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("TRUSS72", x)


def TRUSS200(x: Iterable[float]) -> Tuple[float, float]:
    return run_Engineering("TRUSS200", x)


REFERENCE_SOLUTIONS: Dict[str, np.ndarray] = {
    "WBD": np.array([0.20572964, 3.47048867, 9.03662391, 0.20572964]),
    "CBD": np.array([6.01601590, 5.30917392, 4.49432961, 3.50147609, 2.15266533]),
    "SRD": np.array([3.5, 0.7, 17.0, 7.3, 7.8, 3.35021467, 5.28668323]),
    "RSLD": np.array([99.0956924892341, 0.5963497605675881]),
    "RSLD_COMPLEX": np.array(
        [0.9519119, 0.9464885, 0.9465385, 97.6644, 0.59043213]
    ),
    "ELD": np.array([458.23800, 199.70969, 299.91353, 55.13067, 200.0, 50.0]),
}
