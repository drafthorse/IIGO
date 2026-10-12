"""Crisscross Optimization Algorithm (CSO) in the paper-experiment format.

CSO (Meng et al., Knowledge-Based Systems, 2014) alternates two crossover
operators over the whole population:

* Horizontal crossover (横向交叉): the population is randomly paired and each
  pair (X(i), X(j)) generates two offspring per dimension d

      MShc(i,d) = r1*X(i,d) + (1-r1)*X(j,d) + c1*(X(i,d)-X(j,d))
      MShc(j,d) = r2*X(j,d) + (1-r2)*X(i,d) + c2*(X(j,d)-X(i,d))

  with r1, r2 ~ U(0,1) and expansion coefficients c1, c2 ~ U(-1,1).
  Each offspring competes greedily against its own parent.

* Vertical crossover (纵向交叉): inside a single individual two distinct
  dimensions d1, d2 are blended

      MSvc(i,d1) = r*X(i,d1) + (1-r)*X(i,d2),   r ~ U(0,1)

  applied with probability P_v; the offspring competes against the parent.

The function follows the same contract as every optimizer in
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
def cso(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
        benchmark_id, seed, initial_pop):
    """Crisscross Optimization Algorithm (CSO).

    Recommended parameter values from the original paper: horizontal
    crossover is always performed (P_h = 1.0) and vertical crossover is
    performed with probability P_v = 0.8.
    """
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    p_h = 1.0  # horizontal crossover probability
    p_v = 0.8  # vertical crossover probability
    while fes < max_fes:
        # ---------- Horizontal crossover ----------
        order = np.random.permutation(search_num)
        for pair in range(search_num // 2):
            if fes >= max_fes:
                break
            i = order[2 * pair]
            j = order[2 * pair + 1]
            if np.random.random() >= p_h:
                continue
            child_i = np.empty(dim)
            child_j = np.empty(dim)
            for d in range(dim):
                r1 = np.random.random()
                r2 = np.random.random()
                c1 = 2.0 * np.random.random() - 1.0
                c2 = 2.0 * np.random.random() - 1.0
                child_i[d] = (r1 * pop[i, d] + (1.0 - r1) * pop[j, d]
                              + c1 * (pop[i, d] - pop[j, d]))
                child_j[d] = (r2 * pop[j, d] + (1.0 - r2) * pop[i, d]
                              + c2 * (pop[j, d] - pop[i, d]))
            child_i = _clip_1d(child_i, lb, ub)
            child_j = _clip_1d(child_j, lb, ub)

            # Greedy competition: each offspring challenges its own parent.
            fit_i = _one(child_i, M, o, func_name, s, benchmark_id)
            fes += 1
            if fit_i < fit[i]:
                pop[i] = child_i
                fit[i] = fit_i
                if fit_i < best_value:
                    best_value = fit_i
                    best_pos = child_i.copy()
            if fes >= max_fes:
                break
            fit_j = _one(child_j, M, o, func_name, s, benchmark_id)
            fes += 1
            if fit_j < fit[j]:
                pop[j] = child_j
                fit[j] = fit_j
                if fit_j < best_value:
                    best_value = fit_j
                    best_pos = child_j.copy()

        # ---------- Vertical crossover ----------
        if dim >= 2:
            for i in range(search_num):
                if fes >= max_fes:
                    break
                if np.random.random() >= p_v:
                    continue
                d1 = np.random.randint(dim)
                d2 = np.random.randint(dim)
                while d2 == d1:
                    d2 = np.random.randint(dim)
                r = np.random.random()
                trial = pop[i].copy()
                trial[d1] = r * pop[i, d1] + (1.0 - r) * pop[i, d2]
                trial = _clip_1d(trial, lb, ub)
                trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
                fes += 1
                if trial_fit < fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fit
                    if trial_fit < best_value:
                        best_value = trial_fit
                        best_pos = trial.copy()

        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


def get_f(name):
    functions = {
        "CSO": cso,
    }
    if name not in functions:
        raise ValueError(f"Unsupported paper algorithm: {name}")
    return functions[name]
