"""Shared helpers for the DE-family optimizers in this folder.

Every optimizer built on these helpers follows the common contract of
ExperimentAlgorithms.py:

* the supplied ``initial_pop`` is copied verbatim (via ``_start``);
* ``seed`` controls all stochastic operations after initialization;
* every call to the objective is counted and the last count is ``max_fes``;
* the returned best position and best fitness always refer to the same point.
"""

import math

import numba
import numpy as np

from elements import F


@numba.njit(nogil=True)
def _clip_1d(x, lb, ub):
    for j in range(x.shape[0]):
        if x[j] < lb:
            x[j] = lb
        elif x[j] > ub:
            x[j] = ub
    return x


@numba.njit(nogil=True)
def _evaluate(pop, M, o, func_name, s, benchmark_id):
    return F(pop, M, o, func_name, s, benchmark_id)[:, 0]


@numba.njit(nogil=True)
def _one(x, M, o, func_name, s, benchmark_id):
    return F(x.reshape((1, x.shape[0])), M, o, func_name, s, benchmark_id)[0, 0]


@numba.njit(nogil=True)
def _start(max_fes, initial_pop, M, o, func_name, s, benchmark_id):
    pop = initial_pop.copy()
    fit = _evaluate(pop, M, o, func_name, s, benchmark_id)
    fes = pop.shape[0]
    best_idx = int(np.argmin(fit))
    best_fit = fit[best_idx]
    best_pos = pop[best_idx].copy()
    curve = np.empty(max_fes + 1, dtype=np.float64)
    fe_curve = np.empty(max_fes + 1, dtype=np.int64)
    curve[0] = best_fit
    fe_curve[0] = fes
    return pop, fit, fes, best_pos, best_fit, curve, fe_curve


@numba.njit(nogil=True)
def _cauchy(mu, sigma):
    """Cauchy sample truncated to (0, 1] as in JADE/LSHADE."""
    v = mu + sigma * math.tan(math.pi * (np.random.random() - 0.5))
    while v <= 0.0:
        v = mu + sigma * math.tan(math.pi * (np.random.random() - 0.5))
    if v > 1.0:
        v = 1.0
    return v


@numba.njit(nogil=True)
def _matlab_round(x):
    """Round half away from zero (MATLAB ``round``) for positive x."""
    return int(math.floor(x + 0.5))
