"""jSO: improved iL-SHADE, CEC2017 runner-up.

Reference: J. Brest, M. S. Maucec and B. Boskovic, "Single objective
real-parameter optimization: Algorithm jSO," Proc. IEEE Congr. Evol.
Comput. (CEC), pp. 1311-1318, 2017. (IEEE Xplore document 7969456)

This is a faithful port of the authors' official C++ implementation
(jSO-SOURCE-RESULTS-29May2017, lshade.cc), adapted to the shared experiment
contract of ExperimentAlgorithms.py:

* the supplied ``initial_pop`` is copied verbatim and ``search_num`` plays the
  role of the initial population size (the official code uses
  round(sqrt(D)*ln(D)*25); here the experiment protocol fixes the shared
  initial population);
* ``seed`` controls all stochastic operations after initialization;
* every call to the objective is counted and the last count is ``max_fes``;
* the returned best position and best fitness always refer to the same point;
* the official CEC error-clamping rule (error < 1e-8 is reported as 0) is a
  reporting convention that requires the known optimum; it is omitted here so
  that the returned best fitness always matches a re-evaluation.

Official jSO mechanisms beyond L-SHADE (progress tau = nfes/max_nfes):

* memory M_F initialized to 0.3 and M_CR to 0.8 (L-SHADE used 0.5/0.5);
* the last memory slot is locked: when it is selected, mu_F = mu_CR = 0.9;
* weighted mutation v = x + Fw*F*(x_pbest - x) + F*(x_r1 - x_r2~) with
  Fw = 0.7 for tau < 0.2, 0.8 for tau < 0.4, else 1.2;
* CR clamping: CR >= 0.7 for tau < 0.25 and CR >= 0.6 for tau < 0.5;
* F clamping: F <= 0.7 for tau < 0.6;
* p_best_rate = 0.25 decreased linearly to half over the run
  (updated together with the population reduction step);
* historical memory entries are updated with the mean of the new weighted
  Lehmer mean and the previous entry value;
* pbest is redrawn while it equals the target for tau < 0.5;
* trial vectors with fitness equal to the parent replace it without entering
  the success history or the archive.

Other official settings: memory_size = 5, arc_rate = 1.0, min_pop_size = 4,
midpoint bound handling (parent medium).

"""

import math

import numba
import numpy as np

from de_common import _cauchy, _matlab_round, _one, _start


@numba.njit(nogil=True)
def jso(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
        benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1

    # Official parameter settings.
    memory_size = 5
    arc_rate = 1.0
    p_best_rate0 = 0.25
    min_pop = 4
    max_pop = search_num
    current_size = search_num

    memory_sf = np.full(memory_size, 0.3)   # jSO
    memory_cr = np.full(memory_size, 0.8)   # jSO
    memory_pos = 0

    p_best_rate = p_best_rate0
    p_num = max(1, _matlab_round(current_size * p_best_rate))

    arc_max = max(1, _matlab_round(current_size * arc_rate))
    archive = np.empty((arc_max, dim))
    archive_count = 0

    while fes < max_fes:
        old_pop = pop[:current_size].copy()
        old_fit = fit[:current_size].copy()
        order = np.argsort(old_fit)
        batch = min(current_size, max_fes - fes)
        tau = fes / max_fes  # progress at generation start (official nfes)

        v_arr = np.empty((batch, dim))
        used_f = np.empty(batch)
        used_cr = np.empty(batch)
        for i in range(batch):
            mem_idx = np.random.randint(memory_size)
            if mem_idx == memory_size - 1:
                mu_sf = 0.9  # locked last memory slot
                mu_cr = 0.9
            else:
                mu_sf = memory_sf[mem_idx]
                mu_cr = memory_cr[mem_idx]

            # CR with jSO clamping.
            if mu_cr < 0.0:
                cri = 0.0
            else:
                cri = mu_cr + 0.1 * np.random.normal()
                cri = min(1.0, max(0.0, cri))
            if tau < 0.25 and cri < 0.7:
                cri = 0.7
            if tau < 0.5 and cri < 0.6:
                cri = 0.6

            # F with jSO clamping.
            fi = _cauchy(mu_sf, 0.1)
            if tau < 0.6 and fi > 0.7:
                fi = 0.7
            used_f[i] = fi
            used_cr[i] = cri

            # pbest, redrawn while equal to the target during the first half.
            pbest_idx = order[np.random.randint(p_num)]
            while tau < 0.5 and pbest_idx == i:
                pbest_idx = order[np.random.randint(p_num)]

            # Weighted scaling factor for the pbest difference term.
            if tau < 0.2:
                jf = fi * 0.7
            elif tau < 0.4:
                jf = fi * 0.8
            else:
                jf = fi * 1.2

            r1 = np.random.randint(current_size)
            while r1 == i:
                r1 = np.random.randint(current_size)
            extended = current_size + archive_count
            r2 = np.random.randint(extended)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(extended)
            if r2 < current_size:
                xr2 = old_pop[r2]
            else:
                xr2 = archive[r2 - current_size]

            # current-to-pbest-w/1 mutation + binomial crossover.
            trial = old_pop[i].copy()
            jrand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < cri or j == jrand:
                    trial[j] = (old_pop[i, j]
                                + jf * (old_pop[pbest_idx, j] - old_pop[i, j])
                                + fi * (old_pop[r1, j] - xr2[j]))
            # Midpoint bound handling (parent medium).
            for j in range(dim):
                if trial[j] < lb:
                    trial[j] = (old_pop[i, j] + lb) / 2.0
                elif trial[j] > ub:
                    trial[j] = (old_pop[i, j] + ub) / 2.0
            v_arr[i] = trial

        # Evaluation and generation alternation.
        succ_f = np.empty(batch)
        succ_cr = np.empty(batch)
        succ_dif = np.empty(batch)
        success = 0
        for i in range(batch):
            trial_fit = _one(v_arr[i], M, o, func_name, s, benchmark_id)
            fes += 1
            if trial_fit < best_value:
                best_value = trial_fit
                best_pos = v_arr[i].copy()
            if trial_fit < old_fit[i]:
                if archive_count < arc_max:
                    archive[archive_count] = old_pop[i]
                    archive_count += 1
                else:
                    archive[np.random.randint(arc_max)] = old_pop[i]
                succ_f[success] = used_f[i]
                succ_cr[success] = used_cr[i]
                succ_dif[success] = abs(old_fit[i] - trial_fit)
                success += 1
                pop[i] = v_arr[i]
                fit[i] = trial_fit
            elif trial_fit == old_fit[i]:
                # Equal-fitness trials replace the parent silently (jSO).
                pop[i] = v_arr[i]
                fit[i] = trial_fit

        # Success-history update: weighted Lehmer mean averaged with the
        # previous memory entry (jSO).
        if success > 0:
            sum_dif = 0.0
            for k in range(success):
                sum_dif += succ_dif[k]
            if sum_dif > 0.0:
                old_sf = memory_sf[memory_pos]
                old_cr = memory_cr[memory_pos]
                numer_f = 0.0
                denom_f = 0.0
                numer_cr = 0.0
                denom_cr = 0.0
                for k in range(success):
                    w = succ_dif[k] / sum_dif
                    numer_f += w * succ_f[k] * succ_f[k]
                    denom_f += w * succ_f[k]
                    numer_cr += w * succ_cr[k] * succ_cr[k]
                    denom_cr += w * succ_cr[k]
                new_sf = numer_f / denom_f if denom_f > 0.0 else old_sf
                if denom_cr == 0.0:
                    new_cr = -1.0
                else:
                    new_cr = numer_cr / denom_cr
                memory_sf[memory_pos] = (new_sf + old_sf) / 2.0
                memory_cr[memory_pos] = (new_cr + old_cr) / 2.0
                memory_pos = (memory_pos + 1) % memory_size

        # Linear population size reduction.
        plan = _matlab_round(((min_pop - max_pop) / max_fes) * fes + max_pop)
        if current_size > plan:
            red = current_size - plan
            if current_size - red < min_pop:
                red = current_size - min_pop
            new_size = current_size - red
            order2 = np.argsort(fit[:current_size])
            new_pop = np.empty((new_size, dim))
            new_fit = np.empty(new_size)
            for k in range(new_size):
                new_pop[k] = pop[order2[k]]
                new_fit[k] = fit[order2[k]]
            pop[:new_size] = new_pop
            fit[:new_size] = new_fit
            current_size = new_size

            # Resize the archive (official code truncates the count).
            arc_max = max(1, int(current_size * arc_rate))
            if archive_count > arc_max:
                archive_count = arc_max

            # jSO: p_best_rate decreases linearly to half of its initial
            # value over the run (updated with the reduction step).
            p_best_rate = p_best_rate0 * (1.0 - 0.5 * fes / max_fes)
            p_num = _matlab_round(current_size * p_best_rate)
            if p_num <= 1:
                p_num = 2

        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


def get_f(name):
    functions = {
        "JSO": jso,
        "jSO": jso,
    }
    if name not in functions:
        raise ValueError(f"Unsupported paper algorithm: {name}")
    return functions[name]
