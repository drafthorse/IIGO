"""LSHADE-cnEpSin: Ensemble Sinusoidal Differential Covariance Matrix
Adaptation with Euclidean Neighborhood.

Reference: N. H. Awad, M. Z. Ali and P. N. Suganthan, "Ensemble sinusoidal
differential covariance matrix adaptation with Euclidean neighborhood for
solving CEC2017 benchmark problems," Proc. IEEE Congr. Evol. Comput. (CEC),
2017.

This is a faithful port of the authors' official MATLAB implementation
(LSHADE_cnEpSin.m, CEC2017 competition code), adapted to the shared
experiment contract of ExperimentAlgorithms.py:

* the supplied ``initial_pop`` is copied verbatim and ``search_num`` plays the
  role of the initial population size (the official code uses 18*D; here the
  experiment protocol fixes the shared initial population);
* ``seed`` controls all stochastic operations after initialization;
* every call to the objective is counted and the last count is ``max_fes``;
* the returned best position and best fitness always refer to the same point;
* G_Max (used by the sinusoidal schedules) is recomputed by simulating the
  linear population-size reduction schedule for the given budget, replacing
  the hard-coded per-dimension constants of the official code.

Official parameter values: p_best_rate = 0.11, arc_rate = 1.4,
memory_size = 5, min_pop_size = 4, freq_init = 0.5, p_b = 0.4 (covariance
crossover probability), p_s = 0.5 (neighborhood fraction), LP = 20
(learning-period generations), sinusoidal-ensemble phase during the first
half of the evaluation budget.

"""

import math

import numba
import numpy as np

from de_common import _cauchy, _matlab_round, _one, _start


@numba.njit(nogil=True)
def _simulate_gmax(max_fes, init_pop, min_pop):
    """Number of generations produced by the LPSR schedule (no objective
    calls), mirroring the hard-coded G_Max table of the official code."""
    g = 0
    fes = init_pop
    pop_size = init_pop
    while fes < max_fes:
        g += 1
        fes += pop_size
        plan = int(math.floor(((min_pop - init_pop) / max_fes) * fes
                              + init_pop + 0.5))
        if pop_size > plan:
            red = pop_size - plan
            if pop_size - red < min_pop:
                red = pop_size - min_pop
            pop_size -= red
    return g


@numba.njit(nogil=True)
def lshade_cnepsin(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
                   benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1

    # Official parameter settings.
    freq_init = 0.5
    p_b = 0.4            # probability of the covariance-matrix crossover
    p_s = 0.5            # Euclidean-neighborhood fraction
    p_best_rate = 0.11
    arc_rate = 1.4
    memory_size = 5
    min_pop = 4
    lp = 20              # learning period (generations)

    max_pop = search_num
    current_size = search_num
    sel = max(2, _matlab_round(p_s * current_size))
    g_max = _simulate_gmax(max_fes, max_pop, min_pop)

    memory_sf = np.full(memory_size, 0.5)
    memory_cr = np.full(memory_size, 0.5)
    memory_freq = np.full(memory_size, freq_init)
    memory_pos = 0

    arc_max = max(1, _matlab_round(arc_rate * max_pop))
    archive = np.empty((arc_max, dim))
    archive_count = 0

    # Per-generation success/failure counters of the two sinusoidal
    # configurations (official goodF1all/badF1all/goodF2all/badF2all).
    good1 = np.zeros(g_max + 2)
    bad1 = np.zeros(g_max + 2)
    good2 = np.zeros(g_max + 2)
    bad2 = np.zeros(g_max + 2)

    gg = 0  # generation counter
    while fes < max_fes:
        gg += 1
        old_pop = pop[:current_size].copy()
        old_fit = fit[:current_size].copy()
        order = np.argsort(old_fit)
        batch = min(current_size, max_fes - fes)

        # ---- Decide the scaling-factor source for this generation --------
        flag1 = False  # non-adaptive sinusoidal decreasing
        flag2 = False  # adaptive sinusoidal increasing (Cauchy frequency)
        sin_scalar = 0.0
        if fes <= max_fes // 2:
            if gg <= lp:
                # Learning period: both configurations with probability 0.5.
                if np.random.random() < 0.5:
                    flag1 = True
                else:
                    flag2 = True
            else:
                # Success-rate based probabilities over the last LP
                # generations (the 1.0 entries guard against NaN, as in the
                # official code).
                ns1 = 0.0
                nf1 = 0.0
                ns2 = 0.0
                nf2 = 0.0
                for hh in range(gg - lp - 1, gg - 1):
                    ns1 += good1[hh]
                    nf1 += bad1[hh]
                    ns2 += good2[hh]
                    nf2 += bad2[hh]
                sum_s1 = ns1 / (ns1 + nf1) + 0.01
                sum_s2 = ns2 / (ns2 + nf2) + 0.01
                p1 = sum_s1 / (sum_s1 + sum_s2)
                p2 = sum_s2 / (sum_s1 + sum_s2)
                if p1 > p2:
                    flag1 = True
                else:
                    flag2 = True
            if flag1:
                sin_scalar = 0.5 * (math.sin(2.0 * math.pi * freq_init * gg
                                             + math.pi)
                                    * ((g_max - gg) / g_max) + 1.0)

        # ---- Mutation: DE/current-to-pbest/1 with archive ----------------
        v_arr = np.empty((batch, dim))
        used_f = np.empty(batch)
        used_cr = np.empty(batch)
        used_freq = np.empty(batch)
        pnp = max(2, _matlab_round(p_best_rate * current_size))
        extended = current_size + archive_count
        for i in range(batch):
            mem_idx = np.random.randint(memory_size)
            mu_sf = memory_sf[mem_idx]
            mu_cr = memory_cr[mem_idx]
            mu_freq = memory_freq[mem_idx]

            if mu_cr == -1.0:
                cri = 0.0
            else:
                cri = mu_cr + 0.1 * np.random.normal()
                cri = min(1.0, max(0.0, cri))
            fi = _cauchy(mu_sf, 0.1)
            freqi = _cauchy(mu_freq, 0.1)
            if flag1:
                fi = sin_scalar
            elif flag2:
                fi = 0.5 * (math.sin(2.0 * math.pi * freqi * gg)
                            * (gg / g_max) + 1.0)
            used_f[i] = fi
            used_cr[i] = cri
            used_freq[i] = freqi

            pbest = old_pop[order[np.random.randint(pnp)]]
            r1 = np.random.randint(current_size)
            while r1 == i:
                r1 = np.random.randint(current_size)
            r2 = np.random.randint(extended)
            while r2 < current_size and (r2 == i or r2 == r1):
                r2 = np.random.randint(extended)
            if r2 < current_size:
                xr2 = old_pop[r2]
            else:
                xr2 = archive[r2 - current_size]

            vi = old_pop[i] + fi * (pbest - old_pop[i] + old_pop[r1] - xr2)
            # Midpoint bound handling (official boundConstraint.m).
            for j in range(dim):
                if vi[j] < lb:
                    vi[j] = (old_pop[i, j] + lb) / 2.0
                elif vi[j] > ub:
                    vi[j] = (old_pop[i, j] + ub) / 2.0
            v_arr[i] = vi

        # ---- Crossover ---------------------------------------------------
        ui = old_pop[:batch].copy()
        use_eigen = np.random.random() < p_b
        if use_eigen:
            # Covariance matrix learning with Euclidean neighborhood
            # (CMLwithEN): the SEL individuals closest to the best one.
            best_ind = old_pop[order[0]]
            dist = np.zeros(current_size)
            for i2 in range(current_size):
                acc = 0.0
                for j in range(dim):
                    d = old_pop[i2, j] - best_ind[j]
                    acc += d * d
                dist[i2] = math.sqrt(acc)
            didx = np.argsort(dist)
            xsel = np.empty((sel, dim))
            for k in range(sel):
                xsel[k] = old_pop[didx[k]]
            xmean = np.zeros(dim)
            for k in range(sel):
                xmean += xsel[k]
            xmean /= sel
            xc = xsel - xmean
            cov = (xc.T @ xc) / (sel - 1)
            cov = 0.5 * (cov + cov.T)
            eigval, rot = np.linalg.eigh(cov)
            # Limit the condition number to 1e20 (official guard).
            if eigval[dim - 1] > 1e20 * eigval[0]:
                tmp = eigval[dim - 1] / 1e20 - eigval[0]
                for j in range(dim):
                    cov[j, j] += tmp
                eigval, rot = np.linalg.eigh(cov)
            rot_t = np.ascontiguousarray(rot.T)
            for i in range(batch):
                xr = old_pop[i] @ rot
                vr = v_arr[i] @ rot
                ur = xr.copy()
                jrand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < used_cr[i] or j == jrand:
                        ur[j] = vr[j]
                ui[i] = ur @ rot_t
        else:
            for i in range(batch):
                jrand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < used_cr[i] or j == jrand:
                        ui[i, j] = v_arr[i, j]

        # ---- Evaluation, selection, archive ------------------------------
        succ_f = np.empty(batch)
        succ_cr = np.empty(batch)
        succ_freq = np.empty(batch)
        succ_dif = np.empty(batch)
        success = 0
        for i in range(batch):
            trial_fit = _one(ui[i], M, o, func_name, s, benchmark_id)
            fes += 1
            if trial_fit < best_value:
                best_value = trial_fit
                best_pos = ui[i].copy()
            if old_fit[i] > trial_fit:
                if arc_max > 0:
                    if archive_count < arc_max:
                        archive[archive_count] = old_pop[i]
                        archive_count += 1
                    else:
                        archive[np.random.randint(arc_max)] = old_pop[i]
                pop[i] = ui[i]
                fit[i] = trial_fit
                succ_f[success] = used_f[i]
                succ_cr[success] = used_cr[i]
                succ_freq[success] = used_freq[i]
                succ_dif[success] = abs(old_fit[i] - trial_fit)
                success += 1

        # Per-generation success/failure counts of the sinusoidal configs.
        if flag1:
            good1[gg - 1] = success
            bad1[gg - 1] = batch - success
            good2[gg - 1] = 1.0
            bad2[gg - 1] = 1.0
        elif flag2:
            good2[gg - 1] = success
            bad2[gg - 1] = batch - success
            good1[gg - 1] = 1.0
            bad1[gg - 1] = 1.0

        # ---- Success-history memory update (weighted Lehmer means) -------
        if success > 0:
            sum_dif = 0.0
            for k in range(success):
                sum_dif += succ_dif[k]
            if sum_dif > 0.0:
                denom = 0.0
                numer = 0.0
                for k in range(success):
                    w = succ_dif[k] / sum_dif
                    numer += w * succ_f[k] * succ_f[k]
                    denom += w * succ_f[k]
                if denom > 0.0:
                    memory_sf[memory_pos] = numer / denom

                max_cr = 0.0
                numer = 0.0
                denom = 0.0
                for k in range(success):
                    if succ_cr[k] > max_cr:
                        max_cr = succ_cr[k]
                    w = succ_dif[k] / sum_dif
                    numer += w * succ_cr[k] * succ_cr[k]
                    denom += w * succ_cr[k]
                if max_cr == 0.0 or memory_cr[memory_pos] == -1.0:
                    memory_cr[memory_pos] = -1.0
                elif denom > 0.0:
                    memory_cr[memory_pos] = numer / denom

                max_freq = 0.0
                numer = 0.0
                denom = 0.0
                for k in range(success):
                    if succ_freq[k] > max_freq:
                        max_freq = succ_freq[k]
                    w = succ_dif[k] / sum_dif
                    numer += w * succ_freq[k] * succ_freq[k]
                    denom += w * succ_freq[k]
                if max_freq == 0.0 or memory_freq[memory_pos] == -1.0:
                    memory_freq[memory_pos] = -1.0
                elif denom > 0.0:
                    memory_freq[memory_pos] = numer / denom

                memory_pos = (memory_pos + 1) % memory_size

        # ---- Linear population size reduction (LPSR) ---------------------
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
            sel = max(2, _matlab_round(p_s * current_size))
            arc_max = max(1, _matlab_round(arc_rate * current_size))
            if archive_count > arc_max:
                keep = np.random.permutation(archive_count)
                tmp_arc = archive[:arc_max].copy()
                for k in range(arc_max):
                    tmp_arc[k] = archive[keep[k]]
                archive[:arc_max] = tmp_arc
                archive_count = arc_max

        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


def get_f(name):
    functions = {
        "LSHADE-CNEPSIN": lshade_cnepsin,
        "LSHADE_cnEpSin": lshade_cnepsin,
    }
    if name not in functions:
        raise ValueError(f"Unsupported paper algorithm: {name}")
    return functions[name]
