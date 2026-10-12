"""Optimizer implementations with a shared evaluation-budget interface.

Every optimizer in this module follows one common contract:

* the supplied ``initial_pop`` is copied verbatim;
* ``seed`` controls all stochastic operations after initialization;
* every call to the objective is counted and the last count is ``max_fes``;
* the returned best position and best fitness always refer to the same point.
"""

import math

import numba
import numpy as np

from elements import F, calculate_trend_slope_numba, decaysin, half_pivot
from IIGO_Final import iigo_final
from CSO import cso
from LSHADE_cnEpSin import lshade_cnepsin
from jSO import jso


# Algorithm control parameters. Initial population sizes are set separately
# by the experiment configuration.
PSO_INERTIA = 1.0
PSO_COGNITIVE = 2.0
PSO_SOCIAL = 2.0
PSO_VMAX = 2.0
DE_SCALE = 0.5
DE_CROSSOVER = 0.9
RSA_ALPHA = 0.1
RSA_BETA = 0.005
SMA_RESTART_PROBABILITY = 0.03
SNAKE_Q_THRESHOLD = 0.25
SNAKE_T_THRESHOLD = 0.6
SNAKE_C1 = 0.5
SNAKE_C2 = 0.05
SNAKE_C3 = 2.0
RDWOA_SPIRAL_B = 1.0
WEMFO_SPIRAL_B = 1.0
GCHHO_GAUSSIAN_MEAN = 0.0
GCHHO_GAUSSIAN_STD = 1.0
CMA_ES_SIGMA_FRACTION = 0.3


@numba.njit(nogil=True)
def _clip_1d(x, lb, ub):
    for j in range(x.shape[0]):
        if x[j] < lb:
            x[j] = lb
        elif x[j] > ub:
            x[j] = ub
    return x


@numba.njit(nogil=True)
def _clip_2d(x, lb, ub):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] < lb:
                x[i, j] = lb
            elif x[i, j] > ub:
                x[i, j] = ub
    return x


@numba.njit(nogil=True)
def _reflect_1d(x, lb, ub):
    """Repair bounds by reflection instead of absorbing points at an edge."""
    span = ub-lb
    if span <= 0.0:
        return _clip_1d(x, lb, ub)
    period = 2.0*span
    for j in range(x.shape[0]):
        shifted = (x[j]-lb) % period
        if shifted <= span:
            x[j] = lb+shifted
        else:
            x[j] = ub-(shifted-span)
    return x


@numba.njit(nogil=True)
def _centered_volume_diversity(pop, samples=9):
    """Translation-invariant median normalized volume from sampled triples."""
    n, dim = pop.shape
    if n < 3 or dim < 3:
        return 0.0
    center = _population_mean(pop)
    values = np.empty(samples)
    for sample in range(samples):
        i0 = np.random.randint(n)
        i1 = np.random.randint(n)
        while i1 == i0:
            i1 = np.random.randint(n)
        i2 = np.random.randint(n)
        while i2 == i0 or i2 == i1:
            i2 = np.random.randint(n)

        d0 = np.random.randint(dim)
        d1 = np.random.randint(dim)
        while d1 == d0:
            d1 = np.random.randint(dim)
        d2 = np.random.randint(dim)
        while d2 == d0 or d2 == d1:
            d2 = np.random.randint(dim)

        a00 = pop[i0, d0]-center[d0]
        a01 = pop[i0, d1]-center[d1]
        a02 = pop[i0, d2]-center[d2]
        a10 = pop[i1, d0]-center[d0]
        a11 = pop[i1, d1]-center[d1]
        a12 = pop[i1, d2]-center[d2]
        a20 = pop[i2, d0]-center[d0]
        a21 = pop[i2, d1]-center[d1]
        a22 = pop[i2, d2]-center[d2]
        determinant = (
            a00*(a11*a22-a12*a21)
            - a01*(a10*a22-a12*a20)
            + a02*(a10*a21-a11*a20)
        )
        norm0 = math.sqrt(a00*a00+a01*a01+a02*a02)
        norm1 = math.sqrt(a10*a10+a11*a11+a12*a12)
        norm2 = math.sqrt(a20*a20+a21*a21+a22*a22)
        denominator = norm0*norm1*norm2
        values[sample] = 0.0 if denominator <= 1e-30 else abs(determinant)/denominator
    return np.median(values)


@numba.njit(nogil=True)
def _normalized_window_slope(window):
    """Scale-free least-squares slope used with a small deadband."""
    n = window.shape[0]
    if n < 2:
        return 0.0
    x_mean = 0.5*(n-1)
    y_mean = np.mean(window)
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        dx = i-x_mean
        numerator += dx*(window[i]-y_mean)
        denominator += dx*dx
    scale = max(abs(y_mean), 1e-12)
    return 0.0 if denominator == 0.0 else numerator/(denominator*scale)


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
def _update_best(pop, fit, best_pos, best_fit):
    idx = int(np.argmin(fit))
    if fit[idx] < best_fit:
        best_fit = fit[idx]
        best_pos = pop[idx].copy()
    return best_pos, best_fit


@numba.njit(nogil=True)
def _safe_exp(value):
    if value > 60.0:
        value = 60.0
    elif value < -60.0:
        value = -60.0
    return math.exp(value)


@numba.njit(nogil=True)
def _population_mean(pop):
    mean = np.zeros(pop.shape[1])
    for i in range(pop.shape[0]):
        mean += pop[i]
    return mean/pop.shape[0]


def iigo(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
         benchmark_id, seed, initial_pop):
    """Run IIGO with the supplied initial population and evaluation budget."""
    return iigo_final(
        max_fes, search_num, lb, ub, dim, M, o, func_name, s,
        benchmark_id, seed, initial_pop)


def get_f(name):
    functions = {
        "IIGO": iigo,
        "IIGO_Final": iigo_final,
        "PSO": pso,
        "GWO": gwo,
        "SMA": sma,
        "SCA": sca,
        "RSA": rsa,
        "DE": de,
        "SNAKE": snake,
        "RDWOA": rdwoa,
        "GCHHO": gchho,
        "LSHADE": lshade,
        "WEMFO": wemfo,
        "CMA-ES": cma_es,
        "CSO": cso,
        "LSHADE_cnEpSin": lshade_cnepsin,
        "jSO": jso,
    }
    if name not in functions:
        raise ValueError(f"Unsupported paper algorithm: {name}")
    return functions[name]


@numba.njit(nogil=True)
def de(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
       benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    # DE/rand/1/bin control parameters.
    cr = DE_CROSSOVER
    scale = DE_SCALE
    while fes < max_fes:
        batch = min(search_num, max_fes - fes)
        old_pop = pop.copy()
        old_fit = fit.copy()
        for i in range(batch):
            a = np.random.randint(search_num)
            while a == i:
                a = np.random.randint(search_num)
            b = np.random.randint(search_num)
            while b == i or b == a:
                b = np.random.randint(search_num)
            c = np.random.randint(search_num)
            while c == i or c == a or c == b:
                c = np.random.randint(search_num)
            trial = old_pop[i].copy()
            start = np.random.randint(dim)
            for k in range(dim):
                j = (start + k) % dim
                if np.random.random() < cr or k == dim - 1:
                    trial[j] = old_pop[c, j] + scale * (old_pop[a, j] - old_pop[b, j])
            trial = _clip_1d(trial, lb, ub)
            trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
            if trial_fit <= old_fit[i]:
                pop[i] = trial
                fit[i] = trial_fit
        fes += batch
        best_pos, best_value = _update_best(pop, fit, best_pos, best_value)
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def pso(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
        benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    velocity = np.random.uniform(-PSO_VMAX, PSO_VMAX, (search_num, dim))
    pbest = pop.copy()
    pbest_fit = fit.copy()
    while fes < max_fes:
        batch = min(search_num, max_fes - fes)
        for i in range(batch):
            velocity[i] = (PSO_INERTIA * velocity[i]
                           + PSO_COGNITIVE * np.random.random() * (pbest[i] - pop[i])
                           + PSO_SOCIAL * np.random.random() * (best_pos - pop[i]))
            velocity[i] = _clip_1d(velocity[i], -PSO_VMAX, PSO_VMAX)
            pop[i] = _clip_1d(pop[i] + velocity[i], lb, ub)
            fit[i] = _one(pop[i], M, o, func_name, s, benchmark_id)
            if fit[i] < pbest_fit[i]:
                pbest_fit[i] = fit[i]
                pbest[i] = pop[i].copy()
            if fit[i] < best_value:
                best_value = fit[i]
                best_pos = pop[i].copy()
        fes += batch
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def gwo(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
        benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    while fes < max_fes:
        order = np.argsort(fit)
        alpha = pop[order[0]].copy()
        beta = pop[order[1]].copy()
        delta = pop[order[2]].copy()
        progress = fes / max_fes
        a = 2.0 * (1.0 - progress)
        batch = min(search_num, max_fes - fes)
        for i in range(batch):
            r1 = np.random.random(); r2 = np.random.random()
            x1 = alpha - (2*a*r1-a) * np.abs(2*r2*alpha-pop[i])
            r1 = np.random.random(); r2 = np.random.random()
            x2 = beta - (2*a*r1-a) * np.abs(2*r2*beta-pop[i])
            r1 = np.random.random(); r2 = np.random.random()
            x3 = delta - (2*a*r1-a) * np.abs(2*r2*delta-pop[i])
            pop[i] = _clip_1d((x1+x2+x3)/3.0, lb, ub)
            fit[i] = _one(pop[i], M, o, func_name, s, benchmark_id)
        fes += batch
        best_pos, best_value = _update_best(pop, fit, best_pos, best_value)
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def sca(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
        benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    while fes < max_fes:
        progress = fes / max_fes
        r1 = 2.0 * (1.0 - progress)
        batch = min(search_num, max_fes - fes)
        for i in range(batch):
            for j in range(dim):
                r2 = 2.0 * math.pi * np.random.random()
                r3 = 2.0 * np.random.random()
                if np.random.random() < 0.5:
                    pop[i, j] += r1 * math.sin(r2) * abs(r3*best_pos[j]-pop[i, j])
                else:
                    pop[i, j] += r1 * math.cos(r2) * abs(r3*best_pos[j]-pop[i, j])
            pop[i] = _clip_1d(pop[i], lb, ub)
            fit[i] = _one(pop[i], M, o, func_name, s, benchmark_id)
            if fit[i] < best_value:
                best_value = fit[i]
                best_pos = pop[i].copy()
        fes += batch
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def rsa(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
        benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    while fes < max_fes:
        progress = fes / max_fes
        batch = min(search_num, max_fes - fes)
        for i in range(batch):
            r1 = np.random.randint(search_num)
            while r1 == i:
                r1 = np.random.randint(search_num)
            r2 = np.random.randint(search_num)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(search_num)
            es = 2.0 * (2.0*np.random.random()-1.0) * (1.0-progress)
            mean_i = np.sum(pop[i]) / dim
            trial = pop[i].copy()
            for j in range(dim):
                p = RSA_ALPHA + (pop[i, j]-mean_i)/(best_pos[j]*(ub-lb)+np.spacing(1.0))
                eta = best_pos[j]*p
                rr = (best_pos[j]-pop[r2, j])/(best_pos[j]+np.spacing(1.0))
                if progress <= 0.25:
                    trial[j] = best_pos[j]-eta*RSA_BETA-rr*np.random.random()
                elif progress <= 0.5:
                    trial[j] = best_pos[j]*pop[r1, j]*es*np.random.random()
                elif progress <= 0.75:
                    trial[j] = best_pos[j]*p*np.random.random()
                else:
                    trial[j] = best_pos[j]-eta*np.spacing(1.0)-rr*np.random.random()
            pop[i] = _clip_1d(trial, lb, ub)
            fit[i] = _one(pop[i], M, o, func_name, s, benchmark_id)
            if fit[i] < best_value:
                best_value = fit[i]
                best_pos = pop[i].copy()
        fes += batch
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def sma(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
        benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    weights = np.zeros((search_num, dim))
    while fes < max_fes:
        progress = fes / max_fes
        current_best = np.min(fit)
        current_worst = np.max(fit)
        gap = current_best - current_worst
        if abs(gap) < 1e-15:
            gap = -0.1
        a = np.arctanh(max(1e-12, 1.0-progress))
        b = 1.0-progress
        batch = min(search_num, max_fes-fes)
        old_pop = pop.copy()
        old_fit = fit.copy()
        for i in range(batch):
            r = np.random.random()
            term = (current_best-old_fit[i])/gap + 1.0
            if term < 1e-15:
                term = 1e-15
            if i <= search_num/2:
                weights[i, :] = 1.0+r*math.log10(term)
            else:
                weights[i, :] = 1.0-r*math.log10(term)
            trial = old_pop[i].copy()
            if np.random.random() < SMA_RESTART_PROBABILITY:
                trial = np.random.random(dim)*(ub-lb)+lb
            else:
                p = math.tanh(abs(old_fit[i]-best_value))
                va = 2.0*a*np.random.random(dim)-a
                vc = 2.0*b*np.random.random(dim)-b
                for j in range(dim):
                    ia = np.random.randint(search_num)
                    while ia == i:
                        ia = np.random.randint(search_num)
                    ib = np.random.randint(search_num)
                    while ib == i or ib == ia:
                        ib = np.random.randint(search_num)
                    if np.random.random() < p:
                        trial[j] = best_pos[j]+va[j]*(weights[i, j]*old_pop[ia, j]-old_pop[ib, j])
                    else:
                        trial[j] = vc[j]*old_pop[i, j]
            pop[i] = _clip_1d(trial, lb, ub)
            fit[i] = _one(pop[i], M, o, func_name, s, benchmark_id)
            if fit[i] < best_value:
                best_value = fit[i]
                best_pos = pop[i].copy()
        fes += batch
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def rdwoa(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
          benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    fix = 1.0
    previous_best = best_value
    while fes < max_fes:
        progress = fes/max_fes
        # Random-spare phase.  Fitness is updated for exactly the modified whale.
        batch = min(search_num, max_fes-fes)
        for i in range(batch):
            trial = pop[i].copy()
            for j in range(dim):
                if math.tan((np.random.random()-0.5)*math.pi) < 1.0-progress:
                    trial[j] = best_pos[j]
            trial = _clip_1d(trial, lb, ub)
            trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
            pop[i] = trial
            fit[i] = trial_fit
            if trial_fit < best_value:
                best_value = trial_fit
                best_pos = trial.copy()
        fes += batch
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
        if fes >= max_fes:
            break

        progress = fes/max_fes
        a = 2.0*(1.0-progress)
        batch = min(search_num, max_fes-fes)
        source = pop.copy()
        for i in range(batch):
            r1 = np.random.random(); r2 = np.random.random(); r3 = np.random.random()
            aa = 2.0*a*r1-a
            cc = 2.0*r2
            tw = 1.0-math.tan(math.pi*(r3-0.5)*fix/max_fes)
            w1 = math.pow(max(0.0, 1.0-progress), tw)
            w2 = math.pow(max(0.0, 2.0-2.0*progress), tw)
            if np.random.random() < 0.5:
                if abs(aa) < 1.0:
                    target = best_pos
                else:
                    target = source[np.random.randint(search_num)]
                distance = np.abs(cc*target-source[i])
                if progress > 0.5:
                    trial = target-w2*aa*distance
                else:
                    trial = w1*target-aa*distance
            else:
                distance = np.abs(best_pos-source[i])
                ell = 2.0*np.random.random()-1.0
                spiral = distance*math.cos(2.0*math.pi*ell)*math.exp(RDWOA_SPIRAL_B*ell)
                if progress > 0.5:
                    trial = best_pos+w2*spiral
                else:
                    trial = w1*best_pos+spiral
            trial = _clip_1d(trial, lb, ub)
            trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
            pop[i] = trial
            fit[i] = trial_fit
            if trial_fit < best_value:
                best_value = trial_fit
                best_pos = trial.copy()
        fes += batch
        if previous_best == best_value:
            fix = (fix+1.0)/2.0
        else:
            previous_best = best_value
            fix = 1.0
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def wemfo(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
          benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    order = np.argsort(fit)
    flames = pop[order].copy()
    flame_fit = fit[order].copy()
    fix = 1.0
    previous_best = best_value
    while fes < max_fes:
        progress = fes/max_fes
        flame_num = max(1, int(round(search_num-progress*(search_num-1))))
        batch = min(search_num, max_fes-fes)
        for i in range(batch):
            trial = pop[i].copy()
            rr = -1.0-progress
            for j in range(dim):
                t = np.random.random()*(1.0-rr)+rr
                flame_idx = np.random.randint(flame_num)
                distance = abs(flames[flame_idx, j]-pop[i, j])
                tw = 1.0-math.tan(math.pi*(np.random.random()-0.5)*fix/max_fes)
                if progress > 0.5:
                    weight = math.pow(max(0.0, 2.0-2.0*progress), tw)
                    trial[j] = distance*math.exp(WEMFO_SPIRAL_B*t)*math.cos(2*math.pi*t)+weight*flames[flame_idx, j]
                else:
                    weight = math.pow(max(0.0, 1.0-progress), tw)
                    trial[j] = weight*distance*math.exp(WEMFO_SPIRAL_B*t)*math.cos(2*math.pi*t)+flames[flame_idx, j]
            pop[i] = _clip_1d(trial, lb, ub)
            fit[i] = _one(pop[i], M, o, func_name, s, benchmark_id)
        fes += batch

        # Elitist flame archive: position and fitness are sorted and copied together.
        combined_pop = np.empty((2*search_num, dim))
        combined_fit = np.empty(2*search_num)
        combined_pop[:search_num] = flames
        combined_pop[search_num:] = pop
        combined_fit[:search_num] = flame_fit
        combined_fit[search_num:] = fit
        order = np.argsort(combined_fit)
        for i in range(search_num):
            flames[i] = combined_pop[order[i]].copy()
            flame_fit[i] = combined_fit[order[i]]
        if flame_fit[0] < best_value:
            best_value = flame_fit[0]
            best_pos = flames[0].copy()
        if previous_best == best_value:
            fix = (fix+1.0)/2.0
        else:
            previous_best = best_value
            fix = 1.0
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def snake(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
          benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    male_n = search_num//2
    female_n = search_num-male_n
    male_pop = pop[:male_n].copy(); female_pop = pop[male_n:].copy()
    male_fit = fit[:male_n].copy(); female_fit = fit[male_n:].copy()
    stagnant = 0
    previous_best = best_value
    while fes < max_fes:
        progress = fes/max_fes
        temperature = math.exp(-progress)
        q = SNAKE_C1*math.exp(progress-1.0)
        selected_dims = max(1, int(math.ceil(max(stagnant/max_fes, 0.16)*dim)))
        perm = np.random.permutation(dim)
        batch = min(search_num, max_fes-fes)
        used = 0
        for group in range(2):
            group_n = male_n if group == 0 else female_n
            for unused_i in range(group_n):
                if used >= batch:
                    break
                if group == 0:
                    target = np.random.randint(male_n)
                    trial = male_pop[target].copy()
                    target_fit = male_fit[target]
                else:
                    target = np.random.randint(female_n)
                    trial = female_pop[target].copy()
                    target_fit = female_fit[target]
                if q < SNAKE_Q_THRESHOLD:
                    source = np.random.randint(group_n)
                    ref = np.random.randint(group_n)
                    denom = male_fit[ref] if group == 0 else female_fit[ref]
                    numer = male_fit[source] if group == 0 else female_fit[source]
                    if abs(denom) < 1e-15:
                        denom = 1e-15
                    ability = _safe_exp(-numer/denom)
                    for kk in range(selected_dims):
                        j = perm[kk]
                        base = male_pop[source, j] if group == 0 else female_pop[source, j]
                        trial[j] = base+SNAKE_C2*ability*(np.random.random()*(ub-lb)+lb)
                elif temperature > SNAKE_T_THRESHOLD:
                    sign = 1.0 if np.random.random() > 0.5 else -1.0
                    for kk in range(selected_dims):
                        j = perm[kk]
                        trial[j] = best_pos[j]+sign*SNAKE_C3*temperature*np.random.random()*(best_pos[j]-trial[j])
                else:
                    if group == 0:
                        other_best_idx = int(np.argmin(female_fit))
                        other_best_fit = female_fit[other_best_idx]
                        other_best = female_pop[other_best_idx]
                    else:
                        other_best_idx = int(np.argmin(male_fit))
                        other_best_fit = male_fit[other_best_idx]
                        other_best = male_pop[other_best_idx]
                    denom = target_fit if abs(target_fit) >= 1e-15 else 1e-15
                    factor = _safe_exp(-other_best_fit/denom)
                    for kk in range(selected_dims):
                        j = perm[kk]
                        trial[j] += SNAKE_C3*factor*np.random.random()*(q*other_best[j]-trial[j])
                trial = _clip_1d(trial, lb, ub)
                trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
                # One sampled target index is used consistently for comparison,
                # fitness assignment and individual assignment.
                if trial_fit < target_fit:
                    if group == 0:
                        male_fit[target] = trial_fit
                        male_pop[target] = trial.copy()
                    else:
                        female_fit[target] = trial_fit
                        female_pop[target] = trial.copy()
                if trial_fit < best_value:
                    best_value = trial_fit
                    best_pos = trial.copy()
                used += 1
            if used >= batch:
                break
        fes += used
        if previous_best == best_value:
            stagnant += used
        else:
            stagnant = 0
            previous_best = best_value
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def lshade(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
           benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    memory_size = 6
    p_best_rate = 0.11
    archive_rate = 2.6
    memory_f = np.full(memory_size, 0.5)
    memory_cr = np.full(memory_size, 0.5)
    memory_pos = 0
    archive_capacity = max(1, int(round(archive_rate*search_num)))
    archive = np.empty((archive_capacity, dim))
    archive_limit = archive_capacity
    archive_count = 0
    current_size = search_num
    order = np.argsort(fit)
    pop = pop[order].copy(); fit = fit[order].copy()
    while fes < max_fes:
        batch = min(current_size, max_fes-fes)
        old_pop = pop.copy(); old_fit = fit.copy()
        sf = np.empty(batch); scr = np.empty(batch); delta = np.empty(batch)
        success = 0
        for i in range(batch):
            mem_idx = np.random.randint(memory_size)
            cri = min(1.0, max(0.0, np.random.normal(memory_cr[mem_idx], 0.1)))
            fi = -1.0
            while fi <= 0.0:
                fi = memory_f[mem_idx]+0.1*math.tan(math.pi*(np.random.random()-0.5))
            fi = min(fi, 1.0)
            pbest_size = max(2, int(round(current_size*p_best_rate)))
            pbest_size = min(pbest_size, current_size)
            pbest_idx = np.random.randint(pbest_size)
            r1 = np.random.randint(current_size)
            while r1 == i:
                r1 = np.random.randint(current_size)
            extended = current_size+archive_count
            r2 = np.random.randint(extended)
            while r2 < current_size and (r2 == i or r2 == r1):
                r2 = np.random.randint(extended)
            if r2 < current_size:
                xr2 = old_pop[r2]
            else:
                xr2 = archive[r2-current_size]
            mutant = old_pop[i]+fi*(old_pop[pbest_idx]-old_pop[i])+fi*(old_pop[r1]-xr2)
            trial = old_pop[i].copy()
            jrand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < cri or j == jrand:
                    trial[j] = mutant[j]
            trial = _clip_1d(trial, lb, ub)
            trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
            if trial_fit <= old_fit[i]:
                pop[i] = trial
                fit[i] = trial_fit
                sf[success] = fi; scr[success] = cri
                delta[success] = abs(old_fit[i]-trial_fit)
                success += 1
                if archive_count < archive_limit:
                    archive[archive_count] = old_pop[i]
                    archive_count += 1
                else:
                    archive[np.random.randint(archive_limit)] = old_pop[i]
        fes += batch
        if success > 0:
            total_delta = np.sum(delta[:success])
            if total_delta <= 0.0:
                weights = np.full(success, 1.0/success)
            else:
                weights = delta[:success]/total_delta
            denom = np.sum(weights*sf[:success])
            if denom > 0.0:
                memory_f[memory_pos] = np.sum(weights*sf[:success]*sf[:success])/denom
            memory_cr[memory_pos] = np.sum(weights*scr[:success])
            memory_pos = (memory_pos+1)%memory_size
        order = np.argsort(fit)
        pop = pop[order].copy(); fit = fit[order].copy()
        target_size = int(round(search_num+(4-search_num)*(fes/max_fes)))
        target_size = max(4, min(current_size, target_size))
        if target_size < current_size:
            pop = pop[:target_size].copy()
            fit = fit[:target_size].copy()
            current_size = target_size
        current_archive_capacity = max(1, int(round(archive_rate*current_size)))
        archive_limit = current_archive_capacity
        if archive_count > current_archive_capacity:
            keep = np.random.permutation(archive_count)
            trimmed = np.empty((current_archive_capacity, dim))
            for k in range(current_archive_capacity):
                trimmed[k] = archive[keep[k]]
            archive[:current_archive_capacity] = trimmed
            archive_count = current_archive_capacity
        if fit[0] < best_value:
            best_value = fit[0]
            best_pos = pop[0].copy()
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def gchho(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
          benchmark_id, seed, initial_pop):
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    levy_scale = math.pow(
        math.gamma(2.5)*math.sin(math.pi*0.75)
        /(math.gamma(1.25)*1.5*math.pow(2.0, 0.25)), 1.0/1.5)
    while fes < max_fes:
        for i in range(search_num):
            if fes >= max_fes:
                break
            progress = fes/max_fes
            rabbit = best_pos.copy()
            energy = 2.0*(2.0*np.random.random()-1.0)*(1.0-progress)
            jump = 2.0*(1.0-np.random.random())
            accepted_direct = False
            if abs(energy) >= 1.0:
                if np.random.random() >= 0.5:
                    ridx = np.random.randint(search_num)
                    trial = pop[ridx]-np.random.random()*np.abs(pop[ridx]-2.0*np.random.random()*pop[i])
                else:
                    trial = rabbit-_population_mean(pop)-np.random.random()*(lb+np.random.random()*(ub-lb))
                trial = _clip_1d(trial, lb, ub)
                trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
                fes += 1
                pop[i] = trial; fit[i] = trial_fit
                accepted_direct = True
            else:
                r = np.random.random()
                if abs(energy) > 0.5 and r >= 0.5:
                    trial = rabbit-pop[i]-energy*np.abs(jump*rabbit-pop[i])
                    trial = _clip_1d(trial, lb, ub)
                    trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
                    fes += 1
                    pop[i] = trial; fit[i] = trial_fit
                    accepted_direct = True
                elif abs(energy) <= 0.5 and r < 0.5:
                    trial = rabbit-energy*np.abs(rabbit-pop[i])
                    trial = _clip_1d(trial, lb, ub)
                    trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
                    fes += 1
                    pop[i] = trial; fit[i] = trial_fit
                    accepted_direct = True
                else:
                    if abs(energy) > 0.5:
                        y = rabbit-energy*np.abs(jump*rabbit-pop[i])
                    else:
                        y = rabbit-energy*np.abs(jump*rabbit-_population_mean(pop))
                    y = _clip_1d(y, lb, ub)
                    y_fit = _one(y, M, o, func_name, s, benchmark_id)
                    fes += 1
                    if y_fit < fit[i]:
                        pop[i] = y; fit[i] = y_fit
                    elif fes < max_fes:
                        denom = math.pow(max(np.random.random(), 1e-15), 1.0/1.5)
                        z = y+np.random.random(dim)*0.01*np.random.random()*levy_scale/denom
                        z = _clip_1d(z, lb, ub)
                        z_fit = _one(z, M, o, func_name, s, benchmark_id)
                        fes += 1
                        if z_fit < fit[i]:
                            pop[i] = z; fit[i] = z_fit
            if fit[i] < best_value:
                best_value = fit[i]
                best_pos = pop[i].copy()
            if fes >= max_fes:
                break

            # Gaussian mutation is evaluated once and compared with the exact
            # fitness belonging to hawk i; there is no redundant full-population call.
            operator = np.zeros(dim)
            for j in range(dim):
                if np.random.random() > 0.5:
                    denom = math.pow(max(np.random.random(), 1e-15), 1.0/1.5)
                    operator[j] = np.random.random()*levy_scale/denom
            r1 = np.random.randint(search_num)
            r2 = np.random.randint(search_num)
            while r2 == r1:
                r2 = np.random.randint(search_num)
            normal_r = np.random.normal(GCHHO_GAUSSIAN_MEAN, GCHHO_GAUSSIAN_STD)
            while normal_r < 0.0 or normal_r > 1.0:
                normal_r = np.random.normal(GCHHO_GAUSSIAN_MEAN, GCHHO_GAUSSIAN_STD)
            trial = pop[i]+operator*(pop[r1]-pop[r2])*normal_r
            trial = _clip_1d(trial, lb, ub)
            trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
            fes += 1
            if trial_fit < fit[i]:
                pop[i] = trial; fit[i] = trial_fit
                if trial_fit < best_value:
                    best_value = trial_fit
                    best_pos = trial.copy()
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def iigo_baseline(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
                  benchmark_id, seed, initial_pop):
    """Baseline IIGO variant with explicit evaluation-budget accounting."""
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    current_best = int(np.argmin(fit)); current_worst = int(np.argmax(fit))
    primes = np.array([1, 2, 3, 5, 7], dtype=np.int64)
    tag = -1
    window = np.zeros(4)
    window_point = 0
    trend = 1.0
    sibling_row = np.random.randint(search_num)
    while sibling_row == current_best or sibling_row == current_worst:
        sibling_row = int((sibling_row+primes[np.random.randint(5)])%search_num)
    key = np.random.randint(dim)
    pair = half_pivot(key, dim).astype(np.int64)
    matrix = np.array([
        [pop[current_best, key], pop[current_best, pair[0]], pop[current_best, pair[1]]],
        [pop[sibling_row, key], pop[sibling_row, pair[0]], pop[sibling_row, pair[1]]],
        [pop[current_worst, key], pop[current_worst, pair[0]], pop[current_worst, pair[1]]],
    ])
    numerator = abs(np.linalg.det(matrix))
    denominator = (np.sqrt(np.sum(matrix[0]*matrix[0]))
                   * np.sqrt(np.sum(matrix[1]*matrix[1]))
                   * np.sqrt(np.sum(matrix[2]*matrix[2])))
    lam = 0.0 if denominator == 0.0 else numerator/denominator
    window[0] = lam; window_point = 1
    generation = 1
    total_generations = max(1, int(math.ceil((max_fes-search_num)/search_num)))
    while fes < max_fes:
        # Generation-based phase definition: theta=t/T.
        # FE accounting still controls termination exactly, while the search
        # phases and decay schedule share the same generation-based clock.
        theta = generation/total_generations
        batch = min(search_num, max_fes-fes)
        next_pop = pop.copy()
        for i in range(batch):
            trial = np.zeros(dim)
            sp1 = np.random.randint(search_num)
            sibling = (sp1+1)%search_num if sp1 == i else sp1
            for k in range(dim):
                target = pop[i, k]
                companion = pop[sibling, k]
                decay = decaysin(total_generations, generation)
                if trend > 0.0 or lam == 1.0:
                    sp2 = np.random.randint(dim)
                    direction = (sp2+1)%dim if sp2 == k else sp2
                    if np.random.random() < 0.15+0.25*lam:
                        h = companion-pop[sibling, direction]
                        v = target-companion
                        d = pop[current_best, k]-target
                        trial[k] = (pop[i, direction]+h)+v*math.cos(math.pi*0.5*np.random.random())+0.5*d
                    else:
                        trial[k] = target
                else:
                    if theta > 0.8:
                        distance = np.abs(pop[current_best]-pop[i])
                        ell = 2.0*np.random.random()-1.0
                        trial = pop[current_best]+distance*math.cos(2.0*math.pi*ell)*math.exp(decay*ell)
                    elif theta < 0.3 or lam < 1e-20:
                        trial = np.random.random(dim)*(ub-lb)+lb
                    else:
                        trial = pop[i].copy()
                    window[:] = 0.0
                    window_point = 0
                    trend = 1.0
                    tag = i
                    break
            trial = _clip_1d(trial, lb, ub)
            trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
            if trial_fit <= fit[i]:
                next_pop[i] = trial
                fit[i] = trial_fit
        pop = next_pop
        fes += batch
        current_best = int(np.argmin(fit)); current_worst = int(np.argmax(fit))
        if fit[current_best] < best_value:
            best_value = fit[current_best]
            best_pos = pop[current_best].copy()

        sibling_row = np.random.randint(search_num) if tag == -1 else tag
        while sibling_row == current_best or sibling_row == current_worst:
            sibling_row = int((sibling_row+primes[np.random.randint(5)])%search_num)
        key = np.random.randint(dim)
        pair = half_pivot(key, dim).astype(np.int64)
        matrix = np.array([
            [pop[current_best, key], pop[current_best, pair[0]], pop[current_best, pair[1]]],
            [pop[sibling_row, key], pop[sibling_row, pair[0]], pop[sibling_row, pair[1]]],
            [pop[current_worst, key], pop[current_worst, pair[0]], pop[current_worst, pair[1]]],
        ])
        numerator = abs(np.linalg.det(matrix))
        denominator = (np.sqrt(np.sum(matrix[0]*matrix[0]))
                       * np.sqrt(np.sum(matrix[1]*matrix[1]))
                       * np.sqrt(np.sum(matrix[2]*matrix[2])))
        lam = 0.0 if denominator == 0.0 else numerator/denominator
        if window_point < window.shape[0]:
            window[window_point] = lam
            window_point += 1
        else:
            trend = calculate_trend_slope_numba(window)
            for w in range(window.shape[0]-1):
                window[w] = window[w+1]
            window[-1] = lam
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
        generation += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def iigo_lpsr(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
              benchmark_id, seed, initial_pop):
    """IIGO variant with evaluation-based schedules and population reduction."""
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    initial_size = pop.shape[0]
    min_size = initial_size if initial_size < 10 else 10
    primes = np.array([1, 2, 3, 5, 7], dtype=np.int64)
    window = np.zeros(4)
    window_point = 0
    trend = 1.0

    current_best = int(np.argmin(fit))
    current_worst = int(np.argmax(fit))
    sibling_row = np.random.randint(pop.shape[0])
    while sibling_row == current_best or sibling_row == current_worst:
        sibling_row = int((sibling_row+primes[np.random.randint(5)])%pop.shape[0])
    key = np.random.randint(dim)
    pair = half_pivot(key, dim).astype(np.int64)
    matrix = np.array([
        [pop[current_best, key], pop[current_best, pair[0]], pop[current_best, pair[1]]],
        [pop[sibling_row, key], pop[sibling_row, pair[0]], pop[sibling_row, pair[1]]],
        [pop[current_worst, key], pop[current_worst, pair[0]], pop[current_worst, pair[1]]],
    ])
    numerator = abs(np.linalg.det(matrix))
    denominator = (np.sqrt(np.sum(matrix[0]*matrix[0]))
                   * np.sqrt(np.sum(matrix[1]*matrix[1]))
                   * np.sqrt(np.sum(matrix[2]*matrix[2])))
    lam = 0.0 if denominator == 0.0 else numerator/denominator
    window[0] = lam
    window_point = 1

    while fes < max_fes:
        n = pop.shape[0]
        theta = min(1.0, fes/max_fes)
        decay = decaysin(max_fes, fes)
        batch = min(n, max_fes-fes)
        next_pop = pop.copy()
        tag = -1
        for i in range(batch):
            trial = np.zeros(dim)
            sp1 = np.random.randint(n)
            sibling = (sp1+1)%n if sp1 == i else sp1
            for k in range(dim):
                target = pop[i, k]
                companion = pop[sibling, k]
                if trend > 0.0 or lam == 1.0:
                    sp2 = np.random.randint(dim)
                    direction = (sp2+1)%dim if sp2 == k else sp2
                    if np.random.random() < 0.15+0.25*lam:
                        h = companion-pop[sibling, direction]
                        v = target-companion
                        d = pop[current_best, k]-target
                        trial[k] = ((pop[i, direction]+h)
                                    + v*math.cos(math.pi*0.5*np.random.random())
                                    + 0.5*d)
                    else:
                        trial[k] = target
                else:
                    if theta > 0.8:
                        distance = np.abs(pop[current_best]-pop[i])
                        ell = 2.0*np.random.random()-1.0
                        trial = (pop[current_best]
                                 + distance*math.cos(2.0*math.pi*ell)
                                 * math.exp(decay*ell))
                    elif theta < 0.3 or lam < 1e-20:
                        trial = np.random.random(dim)*(ub-lb)+lb
                    else:
                        trial = pop[i].copy()
                    window[:] = 0.0
                    window_point = 0
                    trend = 1.0
                    tag = i
                    break
            trial = _clip_1d(trial, lb, ub)
            trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
            if trial_fit <= fit[i]:
                next_pop[i] = trial
                fit[i] = trial_fit
        pop = next_pop
        fes += batch

        desired_size = int(math.floor(
            initial_size-(initial_size-min_size)*(fes/max_fes)+0.5))
        desired_size = max(min_size, min(initial_size, desired_size))
        if desired_size < pop.shape[0]:
            survivors = np.argsort(fit)[:desired_size]
            pop = pop[survivors].copy()
            fit = fit[survivors].copy()

        n = pop.shape[0]
        current_best = int(np.argmin(fit))
        current_worst = int(np.argmax(fit))
        if fit[current_best] < best_value:
            best_value = fit[current_best]
            best_pos = pop[current_best].copy()

        sibling_row = np.random.randint(n)
        if tag >= 0 and tag < n:
            sibling_row = tag
        while sibling_row == current_best or sibling_row == current_worst:
            sibling_row = int((sibling_row+primes[np.random.randint(5)])%n)
        key = np.random.randint(dim)
        pair = half_pivot(key, dim).astype(np.int64)
        matrix = np.array([
            [pop[current_best, key], pop[current_best, pair[0]], pop[current_best, pair[1]]],
            [pop[sibling_row, key], pop[sibling_row, pair[0]], pop[sibling_row, pair[1]]],
            [pop[current_worst, key], pop[current_worst, pair[0]], pop[current_worst, pair[1]]],
        ])
        numerator = abs(np.linalg.det(matrix))
        denominator = (np.sqrt(np.sum(matrix[0]*matrix[0]))
                       * np.sqrt(np.sum(matrix[1]*matrix[1]))
                       * np.sqrt(np.sum(matrix[2]*matrix[2])))
        lam = 0.0 if denominator == 0.0 else numerator/denominator
        if window_point < window.shape[0]:
            window[window_point] = lam
            window_point += 1
        else:
            trend = calculate_trend_slope_numba(window)
            for w in range(window.shape[0]-1):
                window[w] = window[w+1]
            window[-1] = lam
        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def _iigo_rank_core(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
                    benchmark_id, seed, initial_pop, variant):
    """Shared implementation of rank-oriented IIGO ensemble variants."""
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    initial_size = pop.shape[0]
    if variant == 1:
        min_size = 6
        iigo_start, iigo_end = 0.65, 0.30
        p_start, p_end = 0.25, 0.08
        restart_fraction, restart_interval = 0.20, 0.08
        reduction_power = 1.0
    elif variant == 2:
        min_size = 4
        iigo_start, iigo_end = 0.50, 0.20
        p_start, p_end = 0.18, 0.05
        restart_fraction, restart_interval = 0.30, 0.05
        reduction_power = 0.70
    else:
        min_size = 8
        iigo_start, iigo_end = 0.75, 0.40
        p_start, p_end = 0.30, 0.10
        restart_fraction, restart_interval = 0.15, 0.10
        reduction_power = 1.30
    min_size = min(initial_size, max(4, min_size))

    memory_size = 6
    memory_f = np.full(memory_size, 0.5)
    memory_cr = np.full(memory_size, 0.8)
    memory_index = 0
    archive = np.empty((initial_size, dim))
    archive_count = 0
    score_iigo = 0.5
    score_de = 0.5
    last_improvement_fes = fes

    while fes < max_fes:
        n = pop.shape[0]
        progress = min(1.0, fes/max_fes)
        order = np.argsort(fit)
        current_best = int(order[0])
        p_rate = p_end+(p_start-p_end)*(1.0-progress)
        p_count = max(2, min(n, int(math.ceil(p_rate*n))))
        base_iigo = iigo_end+(iigo_start-iigo_end)*(1.0-progress)
        adaptive_iigo = score_iigo/max(1e-12, score_iigo+score_de)
        probability_iigo = min(0.85, max(0.10, 0.5*base_iigo+0.5*adaptive_iigo))
        lam = _centered_volume_diversity(pop, 7)

        batch = min(n, max_fes-fes)
        next_pop = pop.copy()
        successful_f = np.empty(batch)
        successful_cr = np.empty(batch)
        success_weight = np.empty(batch)
        success_count = 0
        trials_iigo = 0
        trials_de = 0
        successes_iigo = 0
        successes_de = 0

        for i in range(batch):
            memory_slot = np.random.randint(memory_size)
            f_scale = -1.0
            while f_scale <= 0.0:
                f_scale = (memory_f[memory_slot]
                           + 0.1*math.tan(math.pi*(np.random.random()-0.5)))
            f_scale = min(1.0, f_scale)
            crossover = min(1.0, max(
                0.0, memory_cr[memory_slot]+0.1*np.random.normal(0.0, 1.0)))
            use_iigo = np.random.random() < probability_iigo

            if use_iigo:
                trials_iigo += 1
                sibling = np.random.randint(n)
                while sibling == i:
                    sibling = np.random.randint(n)
                forced = np.random.randint(dim)
                trial = pop[i].copy()
                mutation_probability = min(0.65, 0.15+0.25*lam)
                for k in range(dim):
                    if k == forced or np.random.random() < mutation_probability:
                        direction = np.random.randint(dim)
                        while direction == k:
                            direction = np.random.randint(dim)
                        target = pop[i, k]
                        companion = pop[sibling, k]
                        h = companion-pop[sibling, direction]
                        v = target-companion
                        d = pop[current_best, k]-target
                        trial[k] = ((pop[i, direction]+h)
                                    + v*math.cos(0.5*math.pi*np.random.random())
                                    + 0.5*d)
                if progress > 0.85 and np.random.random() < 0.15:
                    ell = 2.0*np.random.random()-1.0
                    distance = np.abs(pop[current_best]-pop[i])
                    decay = decaysin(max_fes, fes)
                    trial = (pop[current_best]
                             + distance*math.cos(2.0*math.pi*ell)
                             * math.exp(decay*ell))
            else:
                trials_de += 1
                pbest = int(order[np.random.randint(p_count)])
                r1 = np.random.randint(n)
                while r1 == i or r1 == pbest:
                    r1 = np.random.randint(n)
                combined_count = n+archive_count
                selected = np.random.randint(combined_count)
                attempts = 0
                while selected < n and (selected == i or selected == pbest
                                        or selected == r1) and attempts < 20:
                    selected = np.random.randint(combined_count)
                    attempts += 1
                if selected < n:
                    difference_point = pop[selected]
                else:
                    difference_point = archive[selected-n]
                mutant = (pop[i]+f_scale*(pop[pbest]-pop[i])
                          + f_scale*(pop[r1]-difference_point))
                trial = pop[i].copy()
                forced = np.random.randint(dim)
                for k in range(dim):
                    if k == forced or np.random.random() < crossover:
                        trial[k] = mutant[k]

            trial = _clip_1d(trial, lb, ub)
            trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
            if trial_fit <= fit[i]:
                improvement = max(0.0, fit[i]-trial_fit)
                if archive_count < initial_size:
                    archive[archive_count] = pop[i]
                    archive_count += 1
                else:
                    archive[np.random.randint(initial_size)] = pop[i]
                next_pop[i] = trial
                fit[i] = trial_fit
                if use_iigo:
                    successes_iigo += 1
                else:
                    successes_de += 1
                    successful_f[success_count] = f_scale
                    successful_cr[success_count] = crossover
                    success_weight[success_count] = max(improvement, 1e-30)
                    success_count += 1
        pop = next_pop
        fes += batch

        if success_count > 0:
            total_weight = np.sum(success_weight[:success_count])
            lehmer_f_top = 0.0
            lehmer_f_bottom = 0.0
            lehmer_cr_top = 0.0
            lehmer_cr_bottom = 0.0
            for q in range(success_count):
                weight = success_weight[q]/total_weight
                value_f = successful_f[q]
                value_cr = successful_cr[q]
                lehmer_f_top += weight*value_f*value_f
                lehmer_f_bottom += weight*value_f
                lehmer_cr_top += weight*value_cr*value_cr
                lehmer_cr_bottom += weight*value_cr
            if lehmer_f_bottom > 0.0:
                memory_f[memory_index] = lehmer_f_top/lehmer_f_bottom
            if lehmer_cr_bottom > 0.0:
                memory_cr[memory_index] = lehmer_cr_top/lehmer_cr_bottom
            memory_index = (memory_index+1)%memory_size

        rate_iigo = (successes_iigo+0.5)/(trials_iigo+1.0)
        rate_de = (successes_de+0.5)/(trials_de+1.0)
        score_iigo = 0.8*score_iigo+0.2*rate_iigo
        score_de = 0.8*score_de+0.2*rate_de

        best_index = int(np.argmin(fit))
        if fit[best_index] < best_value:
            best_value = fit[best_index]
            best_pos = pop[best_index].copy()
            last_improvement_fes = fes

        reduction_progress = progress**reduction_power
        desired_size = int(math.floor(
            initial_size-(initial_size-min_size)*reduction_progress+0.5))
        desired_size = max(min_size, min(initial_size, desired_size))
        if desired_size < pop.shape[0]:
            survivors = np.argsort(fit)[:desired_size]
            pop = pop[survivors].copy()
            fit = fit[survivors].copy()

        stagnation_limit = max(initial_size, int(restart_interval*max_fes))
        if (fes < max_fes and fes-last_improvement_fes >= stagnation_limit
                and progress < 0.97):
            n = pop.shape[0]
            order = np.argsort(fit)
            restart_count = max(1, int(math.ceil(restart_fraction*n)))
            restart_count = min(restart_count, max_fes-fes)
            spread = np.zeros(dim)
            mean = _population_mean(pop)
            for row in range(n):
                difference = pop[row]-mean
                spread += difference*difference
            spread = np.sqrt(spread/max(1, n-1))
            for q in range(restart_count):
                target_index = int(order[n-1-q])
                if np.random.random() < 0.6:
                    trial = (best_pos+np.random.normal(0.0, 1.0, dim)
                             * spread*(0.25+0.75*(1.0-progress)))
                else:
                    trial = (lb+ub-pop[target_index]
                             + 0.03*(1.0-progress)*(ub-lb)
                             * np.random.normal(0.0, 1.0, dim))
                trial = _clip_1d(trial, lb, ub)
                trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
                pop[target_index] = trial
                fit[target_index] = trial_fit
                fes += 1
                if trial_fit < best_value:
                    best_value = trial_fit
                    best_pos = trial.copy()
            last_improvement_fes = fes

        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


def iigo_rank_a(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
                benchmark_id, seed, initial_pop):
    return _iigo_rank_core(
        max_fes, search_num, lb, ub, dim, M, o, func_name, s,
        benchmark_id, seed, initial_pop, 1)


def iigo_rank_b(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
                benchmark_id, seed, initial_pop):
    return _iigo_rank_core(
        max_fes, search_num, lb, ub, dim, M, o, func_name, s,
        benchmark_id, seed, initial_pop, 2)


def iigo_rank_c(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
                benchmark_id, seed, initial_pop):
    return _iigo_rank_core(
        max_fes, search_num, lb, ub, dim, M, o, func_name, s,
        benchmark_id, seed, initial_pop, 3)


@numba.njit(nogil=True)
def iigo_adaptive(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
                  benchmark_id, seed, initial_pop):
    """Improved IIGO with robust diversity, LPSR and late p-best refinement.

    The initial population is unchanged. For NP=10 the population remains 10;
    for larger initial populations it is linearly reduced to 10. All schedules
    use consumed objective evaluations, making them comparable across NP.
    """
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1
    initial_size = pop.shape[0]
    min_size = initial_size if initial_size < 10 else 10
    window_size = max(4, int(math.ceil(120.0/max(1, initial_size))))
    window = np.zeros(window_size)
    window_count = 0
    decline_streak = 0
    lam = _centered_volume_diversity(pop)

    while fes < max_fes:
        n = pop.shape[0]
        progress = min(1.0, fes/max_fes)
        order = np.argsort(fit)
        current_best = int(order[0])
        current_worst = int(order[-1])
        p_count = max(2, int(math.ceil(0.2*n)))
        batch = min(n, max_fes-fes)
        next_pop = pop.copy()

        recovery_active = decline_streak >= 2 or lam < 0.005
        late_probability = 0.0
        if progress > 0.65:
            late_probability = min(0.65, 0.65*(progress-0.65)/0.35)

        for i in range(batch):
            use_recovery = recovery_active and (
                i == current_worst or np.random.random() < 0.15)
            use_late_refinement = (
                not use_recovery and np.random.random() < late_probability)

            if use_recovery or use_late_refinement:
                pbest = int(order[np.random.randint(p_count)])
                r1 = np.random.randint(n)
                while r1 == i or r1 == pbest:
                    r1 = np.random.randint(n)
                r2 = np.random.randint(n)
                while r2 == i or r2 == pbest or r2 == r1:
                    r2 = np.random.randint(n)
                if use_recovery:
                    scale = 0.5+0.3*(1.0-progress)
                    mutant = (pop[i]+scale*(pop[pbest]-pop[i])
                              + scale*(pop[r1]-pop[r2]))
                    if progress < 0.7:
                        mutant += (0.02*(1.0-progress)*(ub-lb)
                                   * np.random.normal(0.0, 1.0, dim))
                    crossover = 0.9
                else:
                    scale = 0.1+0.5*(1.0-progress)
                    mutant = (pop[i]+scale*(pop[pbest]-pop[i])
                              + scale*(pop[r1]-pop[r2]))
                    crossover = 0.6+0.35*progress
                trial = pop[i].copy()
                forced = np.random.randint(dim)
                for k in range(dim):
                    if k == forced or np.random.random() < crossover:
                        trial[k] = mutant[k]
            else:
                # Prefer a top-20% companion while retaining global sampling.
                if np.random.random() < 0.7:
                    sibling = int(order[np.random.randint(p_count)])
                else:
                    sibling = np.random.randint(n)
                while sibling == i:
                    sibling = np.random.randint(n)
                forced = np.random.randint(dim)
                trial = pop[i].copy()
                mutation_probability = min(0.65, 0.15+0.25*lam)
                decay = decaysin(max_fes, fes)
                for k in range(dim):
                    if k == forced or np.random.random() < mutation_probability:
                        direction = np.random.randint(dim)
                        while direction == k:
                            direction = np.random.randint(dim)
                        h = pop[sibling, k]-pop[sibling, direction]
                        v = pop[i, k]-pop[sibling, k]
                        attraction = pop[current_best, k]-pop[i, k]
                        angle = 0.5*math.pi*np.random.random()
                        trial[k] = ((pop[i, direction]+h)
                                    + v*math.cos(angle)
                                    + (0.35+0.3*decay)*attraction)

            trial = _reflect_1d(trial, lb, ub)
            trial_fit = _one(trial, M, o, func_name, s, benchmark_id)
            if trial_fit <= fit[i]:
                next_pop[i] = trial
                fit[i] = trial_fit

        pop = next_pop
        fes += batch
        best_index = int(np.argmin(fit))
        if fit[best_index] < best_value:
            best_value = fit[best_index]
            best_pos = pop[best_index].copy()

        # LPSR is inactive at NP=10 and only reduces larger populations.
        desired_size = int(math.floor(
            initial_size-(initial_size-min_size)*(fes/max_fes)+0.5))
        desired_size = max(min_size, min(initial_size, desired_size))
        if desired_size < pop.shape[0]:
            survivors = np.argsort(fit)[:desired_size]
            pop = pop[survivors].copy()
            fit = fit[survivors].copy()

        lam = _centered_volume_diversity(pop)
        if window_count < window_size:
            window[window_count] = lam
            window_count += 1
        else:
            slope = _normalized_window_slope(window)
            if slope < -0.03:
                decline_streak += 1
            elif slope > 0.03:
                decline_streak = 0
            elif decline_streak > 0:
                decline_streak -= 1
            for w in range(window_size-1):
                window[w] = window[w+1]
            window[-1] = lam
            if recovery_active:
                decline_streak = 0

        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value


@numba.njit(nogil=True)
def cma_es(max_fes, search_num, lb, ub, dim, M, o, func_name, s,
           benchmark_id, seed, initial_pop):
    """Canonical rank-mu CMA-ES adapted to the shared experiment contract.

    The common initial population is evaluated first and its arithmetic mean is
    used as the initial distribution mean.  Subsequent samples are controlled
    by the same condition/run seed used by the other optimizers.
    """
    np.random.seed(seed)
    pop, fit, fes, best_pos, best_value, curve, fe_curve = _start(
        max_fes, initial_pop, M, o, func_name, s, benchmark_id)
    rec = 1

    population_size = search_num
    mu = max(1, population_size//2)
    weights = np.empty(mu)
    for i in range(mu):
        weights[i] = math.log(mu+0.5)-math.log(i+1.0)
    weights /= np.sum(weights)
    mueff = 1.0/np.sum(weights*weights)

    cc = (4.0+mueff/dim)/(dim+4.0+2.0*mueff/dim)
    cs = (mueff+2.0)/(dim+mueff+5.0)
    c1 = 2.0/((dim+1.3)*(dim+1.3)+mueff)
    cmu = min(1.0-c1,
              2.0*(mueff-2.0+1.0/mueff)/((dim+2.0)*(dim+2.0)+mueff))
    damps = 1.0+2.0*max(0.0, math.sqrt((mueff-1.0)/(dim+1.0))-1.0)+cs
    chi_n = math.sqrt(dim)*(1.0-1.0/(4.0*dim)+1.0/(21.0*dim*dim))

    mean = _population_mean(initial_pop)
    sigma = CMA_ES_SIGMA_FRACTION*(ub-lb)
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    covariance = np.eye(dim)
    eigenvectors = np.eye(dim)
    axis_scales = np.ones(dim)
    invsqrt_covariance = np.eye(dim)
    generation = 0
    last_eigen_update = 0
    eigen_period = max(1, int(1.0/((c1+cmu)*dim*10.0)))

    while fes < max_fes:
        batch = min(population_size, max_fes-fes)
        candidates = np.empty((batch, dim))
        candidate_fit = np.empty(batch)
        for i in range(batch):
            z = np.random.normal(0.0, 1.0, dim)
            y = eigenvectors @ (axis_scales*z)
            candidate = mean+sigma*y
            candidates[i] = _clip_1d(candidate, lb, ub)
            candidate_fit[i] = _one(candidates[i], M, o, func_name, s, benchmark_id)
        fes += batch

        order = np.argsort(candidate_fit)
        selected = min(mu, batch)
        selected_weights = weights[:selected].copy()
        selected_weights /= np.sum(selected_weights)
        selected_mueff = 1.0/np.sum(selected_weights*selected_weights)
        old_mean = mean.copy()
        mean[:] = 0.0
        for i in range(selected):
            mean += selected_weights[i]*candidates[order[i]]
        mean_step = (mean-old_mean)/sigma

        ps = ((1.0-cs)*ps
              + math.sqrt(cs*(2.0-cs)*selected_mueff)
              * (invsqrt_covariance @ mean_step))
        ps_norm = math.sqrt(np.sum(ps*ps))
        normalization = math.sqrt(max(1e-30, 1.0-math.pow(1.0-cs, 2.0*(generation+1))))
        hsig = 1.0 if ps_norm/normalization/chi_n < 1.4+2.0/(dim+1.0) else 0.0
        pc = ((1.0-cc)*pc
              + hsig*math.sqrt(cc*(2.0-cc)*selected_mueff)*mean_step)

        rank_mu = np.zeros((dim, dim))
        for i in range(selected):
            y = (candidates[order[i]]-old_mean)/sigma
            rank_mu += selected_weights[i]*np.outer(y, y)
        covariance = ((1.0-c1-cmu+(1.0-hsig)*c1*cc*(2.0-cc))*covariance
                      + c1*np.outer(pc, pc)+cmu*rank_mu)
        covariance = 0.5*(covariance+covariance.T)
        sigma *= math.exp((cs/damps)*(ps_norm/chi_n-1.0))
        sigma = min(ub-lb, max(1e-14, sigma))

        if candidate_fit[order[0]] < best_value:
            best_value = candidate_fit[order[0]]
            best_pos = candidates[order[0]].copy()

        generation += 1
        if generation-last_eigen_update >= eigen_period:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            for i in range(dim):
                if eigenvalues[i] < 1e-30:
                    eigenvalues[i] = 1e-30
                axis_scales[i] = math.sqrt(eigenvalues[i])
            invsqrt_covariance = (
                eigenvectors @ np.diag(1.0/axis_scales) @ eigenvectors.T)
            last_eigen_update = generation

        curve[rec] = best_value
        fe_curve[rec] = fes
        rec += 1
    return curve[:rec], fe_curve[:rec], best_pos, best_value
