import numpy as np
import numba
import math
from numba import prange
import CEC2014SO_BC
import CEC2017SO_BC
import CEC2022SO_BC
import EngineerBenchmark
from scipy.spatial import distance

'''
@numba.njit(nogil=True)
def conv_F(t, col, val, row_num, m_ind, o, func_name, s, bound):
    num, dim = t.shape
    n = numba.prange(bound, num - bound, 1)
    y = np.zeros(num - 2 * bound)
    for i in n:
        y[i - bound] = CEC.func(ind=func_name, x=t[i, bound: dim - bound], col=col, val=val, num=row_num,
                                         m_ind=m_ind, o=o, s=s)
    return y


@numba.njit(nogil=True)
def conv_F_1d(t, col, val, row_num, m_ind, o, func_name, s, bound):
    dim = t.shape[0]
    return CEC.func(ind=func_name, x=t[bound: dim - bound], col=col, val=val, num=row_num, m_ind=m_ind, o=o,
                             s=s)
'''



@numba.njit(cache=True)
def my_clip_1d(w, lb, ub, bound):
    m = numba.prange(bound, w.shape[0] - bound, 1)
    for j in m:
        if w[j] > ub:
            w[j] = ub
        elif w[j] < lb:
            w[j] = lb


@numba.njit(nogil=True)
def get_rand_0_1():
    r = np.random.random()
    while r == 0:
        r = np.random.random()
    return r


@numba.njit(nogil=True)
def my_min(t):
    num = t.shape[0]
    n = numba.prange(1, num)
    min_fit = t[0, 0]
    min_pos = 0
    max_fit = t[0, 0]
    max_pos = 0
    for i in n:
        if t[i, 0] < min_fit:
            min_fit = t[i, 0]
            min_pos = i
        if t[i, 0] > max_fit:
            max_fit = t[i, 0]
            max_pos = i
    return min_pos, min_fit, max_pos, max_fit


@numba.njit(nogil=True, fastmath=True)
def F(t, M, o, func_name, s, benchmark_id):
    if benchmark_id == EngineerBenchmark.ENGINEERING_BENCHMARK_ID:
        return EngineerBenchmark.evaluate_population_encoded(t, func_name)
    num, dim = t.shape
    n = numba.prange(num)
    y = np.zeros((num, 2))
    for i in n:
        if benchmark_id == 2014:
            y[i, 0] = CEC2014SO_BC.func(func_name, t[i, :], M, o, s)
        elif benchmark_id == 2017:
            y[i, 0] = CEC2017SO_BC.func(func_name, t[i, :], M, o, s)
        else:
            y[i, 0] = CEC2022SO_BC.func(func_name, t[i, :], M, o, s)
        y[i, 1] = i
    return y


@numba.njit(nogil=True, fastmath=True)
def Fcmd(t, M, o, func_name, s, benchmark_id):
    dim = t.shape
    if benchmark_id == EngineerBenchmark.ENGINEERING_BENCHMARK_ID:
        merit, _, _, _ = EngineerBenchmark.evaluate_encoded(func_name, t)
        return merit
    if benchmark_id == 2014:
        return CEC2014SO_BC.func(func_name, t[:], M, o, s)
    if benchmark_id == 2017:
        return CEC2017SO_BC.func(func_name, t[:], M, o, s)
    return CEC2022SO_BC.func(func_name, t[:], M, o, s)


@numba.njit(nogil=True, fastmath=True)
def F_mesh(X, Y, M, o, func_name, s):
    num1, num2 = X.shape

    n1 = numba.prange(num1)
    n2 = numba.prange(num2)
    y = np.zeros(X.shape)
    for i in n1:
        for j in n2:
            t = np.zeros((1, 2))
            t[0, 0] = X[i, j]
            t[0, 1] = Y[i, j]
            y[i, j] = CEC.func(func_name, t[0, :], M, o, s)
    return y


@numba.njit(nogil=True, fastmath=True)
def F_mesh_2d(X, Y, M, o, func_name, s):
    num1 = X.shape[0]
    n1 = numba.prange(num1)
    y = np.zeros(X.shape)
    for i in n1:
        t = np.zeros((1, 2))
        t[0, 0] = X[i]
        t[0, 1] = Y[i]
        y[i] = CEC.func(func_name, t[0, :], M, o, s)
    return y


@numba.njit(nogil=True, fastmath=True)
def my_clip(w, lb, ub):
    n = numba.prange(w.shape[0])
    m = numba.prange(w.shape[1])
    for i in n:
        for j in m:
            if w[i][j] > ub:
                w[i][j] = ub
            elif w[i][j] < lb:
                w[i][j] = lb
    return w


@numba.njit(nogil=True, fastmath=True)
def mod_clip(w, lb, ub):
    n = numba.prange(w.shape[0])
    m = numba.prange(w.shape[1])
    for i in n:
        for j in m:
            if w[i][j] > ub:
                if ub == 0:
                    w[i][j] = ub
                else:
                    w[i][j] = w[i][j] % ub * ub / np.abs(ub)
            elif w[i][j] < lb:
                if lb == 0:
                    w[i][j] = lb
                else:
                    w[i][j] = w[i][j] % lb * lb / np.abs(lb)
    return w


@numba.njit(nogil=True, fastmath=True)
def cal_diff(dim, arr):
    res = []
    for i in range(dim):
        if i not in arr:
            res.append(i)
    return np.array(res)


@numba.njit(nogil=True, fastmath=True)
def cal_diversity(p):
    p_num = p.shape[0]
    p_mean = np.sum(p, axis=0) / p_num
    all_dvst = np.zeros(p.shape[1])
    for i in range(p_num):
        all_dvst = all_dvst + np.square(p[i, :] - p_mean)
    return all_dvst


@numba.njit(nogil=True, fastmath=True)
def detect_outliers_old(p, threshold=3):
    p_num = p.shape[0]
    mean_d = np.sum(p, axis=0) / p_num
    std_d = np.sqrt(np.mean(cal_diversity(p) / p_num))
    outliers = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        z_score = np.sqrt(np.sum(np.square(p[i] - mean_d)) / p.shape[1]) / std_d
        if np.abs(z_score) > threshold:
            outliers[i] = 1
    return outliers


@numba.njit(nogil=True, fastmath=True)
def detect_outliers(data, threshold=3.0):
    num_samples, num_features = data.shape
    means = np.zeros(num_features)
    std_devs = np.zeros(num_features)
    # 计算每个维度的均值和标准差
    for i in range(num_features):
        means[i] = np.mean(data[:, i])
        std_devs[i] = np.std(data[:, i])
        if std_devs[i] == 0:
            al = [0]
            al.pop()
            return al
    # 计算Z-Scores
    z_scores = np.zeros((num_samples, num_features))
    for i in range(num_samples):
        for j in range(num_features):
            z_scores[i, j] = np.abs((data[i, j] - means[j]) / std_devs[j])

    # 计算每个数据点的综合度量
    composite_scores = np.sum(z_scores, axis=1)
    print(composite_scores)
    # 根据阈值标识离群点
    outliers = composite_scores > threshold
    al = list(np.where(outliers)[0])
    return al


@numba.njit(nogil=True, fastmath=True)
def detect_outliers_d(d, threshold=3):
    p_num = d.shape[0]
    std_d = np.sqrt(cal_diversity(d) / p_num)
    mean_d = np.sum(d, axis=0) / p_num
    print(std_d)
    res = np.zeros(d.shape)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            z_score = (d[i, j] - mean_d[j]) / std_d[j]
            print(z_score)
            if np.abs(z_score) > threshold:
                res[i, j] = 1
    return res


@numba.njit(nogil=True, fastmath=True)
def my_sort(x):
    """
    sort the row of matrix by the first element of each row.
    :param x: a two dimensional matrix
    """

    n = x.shape[0]
    return quick_sort(x, 0, n - 1)


@numba.njit(nogil=True, fastmath=True)
def quick_sort(x, l, r):
    if l >= r:
        return x
    pivot = x[l, :].copy()
    low = l
    high = r
    while l < r:
        while l < r and x[r, 0] >= pivot[0]:
            r -= 1
        x[l, :] = x[r, :]
        while l < r and x[l, 0] <= pivot[0]:
            l += 1
        x[r, :] = x[l, :]
    x[r, :] = pivot
    quick_sort(x, low, l - 1)
    quick_sort(x, l + 1, high)
    return x


@numba.njit(nogil=True, fastmath=True)
def get_mpos(p, parray):
    p_num = parray.shape[0]
    sumarray = np.zeros(parray[0].shape)
    count = 0
    for i in range(p_num):
        if p[i] == 1:
            sumarray += parray[i]
            count += 1
        else:
            sumarray -= parray[i]
            count -= 1
    return sumarray, np.abs(count)


@numba.njit(nogil=True, fastmath=True)
def random_p(arr, p):
    footstep = 0
    res = 0
    aim = np.random.random()
    for i in range(len(p)):
        footstep += p[i]
        if aim < footstep:
            res = arr[i]
            return res


@numba.njit(nogil=True, fastmath=True)
def get_inlier_center(p, ou):
    out_num = ou.shape[0]
    sumarray = np.zeros(p.shape[1], dtype=np.float32)
    ni = 0
    for i in range(p.shape[0]):
        if ni < out_num and i == ou[ni]:
            ni += 1
        else:
            sumarray += np.abs(p[i])
    return sumarray / (p.shape[0] - ni - 1)


@numba.njit(nogil=True, fastmath=True)
def mahalanobis_distance(x, mean_vector, inv_cov_matrix):
    diff = x - mean_vector
    return np.sqrt(np.dot(diff, np.dot(inv_cov_matrix, diff)))


@numba.njit(nogil=True, fastmath=True)
def calculate_mean(data):
    num_features = data.shape[1]
    mean_vector = np.zeros(num_features)

    for j in range(num_features):
        mean_vector[j] = np.mean(data[:, j])

    return mean_vector


@numba.njit(nogil=True, fastmath=True)
def detect_outliers_mahalanobis(data, threshold=3.0):
    num_samples, num_features = data.shape
    mean_vector = calculate_mean(data)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    mahalanobis_dist = np.zeros(num_samples)

    for i in range(num_samples):
        mahalanobis_dist[i] = mahalanobis_distance(data[i, :], mean_vector, inv_cov_matrix)

    outliers = mahalanobis_dist > threshold

    outlier_indices = np.where(outliers)[0]
    return outlier_indices


@numba.njit(nogil=True, fastmath=True)
def detect_outliers_111(data, threshold=3.0):
    num_samples, num_features = data.shape

    # 计算均值
    means = np.zeros(num_features)
    for j in range(num_features):
        means[j] = np.sum(data[:, j]) / num_samples

    # 计算平方差
    squared_diff = np.zeros_like(data)
    for i in range(num_samples):
        for j in range(num_features):
            squared_diff[i, j] = (data[i, j] - means[j]) ** 2

    # 计算方差
    variances = np.sum(squared_diff, axis=0) / num_samples

    # 处理可能的除零错误
    if np.any(variances == 0):
        al = [0]
        al.pop()
        return al

    # 计算 Z-Scores
    z_scores = np.abs(data - means) / np.sqrt(variances)

    # 计算综合分数并标识离群点
    composite_scores = np.sum(z_scores, axis=1)
    threshold_percentile = np.percentile(composite_scores, 90)  # 调整为适当的百分位数
    outliers = composite_scores > threshold_percentile
    # 返回离群点的索引
    al = list(np.where(outliers)[0])
    return al


@numba.njit(nogil=True, fastmath=True)
def remove_element(arr, element):
    res = np.zeros(arr.shape[0] - 1, dtype=np.int64)
    count = 0
    for i in range(arr.shape[0]):
        if arr[i] != element:
            res[count] = arr[i]
            count += 1
    return res


@numba.njit(nogil=True, fastmath=True)
def random_dim(dim, pg, size=1):
    dim1 = np.arange(dim)
    pg1 = pg.astype(np.int64)
    pg_sum = np.sum(pg1)
    p_size = np.zeros(size, dtype=np.int32)
    for i in range(size):
        p_dim = np.zeros(dim1.shape[0])
        for j in range(dim1.shape[0]):
            p_dim[j] = pg1[dim1[j]] / pg_sum
        temp = random_p(dim1, p_dim)
        dim1 = remove_element(dim1, temp)
        p_size[i] = temp
        pg_sum -= pg1[int(temp)]
    return p_size


@numba.njit(nogil=True, fastmath=True)
def get_dif_rind(last_rind, rind_pool):
    rind = np.zeros(4, dtype=np.int32)
    rr = 0
    rl = 0
    while rl < 4:
        if rind_pool[rr] in last_rind:
            rr += 1
        else:
            rind[rl] = rind_pool[rr]
            rl += 1
            rr += 1
    return rind


@numba.njit(nogil=True, fastmath=True)
def sgn(e=0):
    if e>0:
        return 1
    elif e<0:
        return -1
    else:
        return 0


@numba.njit(nogil=True, fastmath=True)
def triangle_outer(a, b, c):
    s = (a + b + c) / 2
    K = np.sqrt(s * (s - a) * (s - b) * (s - c))
    if K == 0:
        print('a:', a, ' b:', b, ' c:', c, ' s:', s, ' K:', K)
        return np.inf  # or any other value that makes sense in your context
    R = (a * b * c) / (4 * K)
    return R


@numba.njit(nogil=True, fastmath=True)
def triangle_inner(a, b, c):
    s = (a + b + c) / 2
    K = np.sqrt(s * (s - a) * (s - b) * (s - c))
    r = K / s
    return r


@numba.njit(nogil=True, fastmath=True)
def Select_ind(n_pool):
    n = n_pool.shape[0]
    for i in prange(n):
        if n_pool[i] == -1:
            continue
        else:
            return i
    return -1


@numba.njit(nogil=True, fastmath=True)
def clip_value(x, min_val, max_val):
    """Clip value between min and max"""
    if x < min_val:
        return min_val
    elif x > max_val:
        return max_val
    return x

@numba.njit(nogil=True, fastmath=True)
def weighted_lehmer_mean(list_objects, list_weights):
    """Calculate the weighted Lehmer mean"""
    up = np.sum(list_weights * list_objects ** 2)
    down = np.sum(list_weights * list_objects)
    return up / down if down != 0 else 0.5

@numba.njit(nogil=True, fastmath=True)
def decaysin(iteration, it):
    T = iteration
    A = 0.5
    sigma = 3 / T
    omega = 10 * np.pi / T
    theta = np.pi/2
    x = A * np.exp(-sigma * it) * np.sin(omega * it + theta)
    if x < 0:
        return 0
    else:
        return x

@numba.njit(nogil=True, fastmath=True)
def half_pivot(key, upper):

    if upper == 2:
        res = np.zeros(2)
        if key == 0:
            res[0] = 0
            res[1] = 1
        else:
            res[0] = 1
            res[1] = 0
        return res
    elif upper % 2 == 0:
        hf = upper / 2
        hr = upper / 2-1
    else:
        hf = (upper-1) / 2
        hr = hf
    pivot = (key + hr + 1) % upper
    fr = np.arange(pivot, pivot + np.floor(hf))
    fr = fr % upper
    rear = np.arange((key+1) % upper, (key+1) % upper + np.floor(hr))
    rear = rear % upper
    res = np.zeros(2)
    a = np.random.choice(fr)
    b = np.random.choice(rear)
    res[0] = a
    res[1] = b
    return res


@numba.njit(nogil=True, fastmath=True)
def polyfit_numba(x, y, deg):
    """
    一个与numpy.polyfit功能类似，但兼容Numba JIT编译的多项式拟合函数。
    返回多项式的系数，顺序为从高次到低次。
    """
    # 构造范德蒙矩阵 (Vandermonde matrix)
    mat = np.zeros((x.shape[0], deg + 1))
    for i in range(x.shape[0]):
        for j in range(deg + 1):
            mat[i, j] = x[i] ** (deg - j)  # 注意：这里顺序与原版polyfit一致
    # 使用最小二乘法求解系数
    coeffs = np.linalg.lstsq(mat, y)[0]
    return coeffs


@numba.njit(nogil=True, fastmath=True)
def calculate_trend_slope_numba(diversity_history):
    """
    使用Numba加速的版本计算滑动窗口内diversity值的趋势斜率。
    diversity_history: 一维数组，存储了最近N代的diversity值。
    """
    window_size = len(diversity_history)
    if window_size < 2:
        return 0.0

    # 生成x坐标（例如代数序列）
    x = np.arange(window_size, dtype=np.float64)
    y = np.asarray(diversity_history, dtype=np.float64)

    # 进行1次（线性）多项式拟合
    coeffs = polyfit_numba(x, y, 1)  # coeffs[0]是斜率，coeffs[1]是截距

    return coeffs[0]  # 返回斜率

def callback():
    print('callback 2022')

@numba.njit(nogil=True, fastmath=True)
def calculate_current_diversity(population, n_samples=50, epsilon=1e-12):
    """
    计算当前种群的多样性指标 (current_diversity)。
    通过多次随机采样三个个体和三个维度，计算fz/fm的比值，并返回中位数。

    参数:
        population: numpy数组，形状为 (n_individuals, n_dimensions)，即当前种群。
        n_samples: 采样次数，默认50次。增加次数更稳定但更慢。
        epsilon: 极小值，防止除以零，默认1e-12。

    返回:
        当前种群的多样性中位数值 (float)。值越接近0越共面（收敛），越接近1越不共面（多样）。
    """
    n, d = population.shape
    # 如果种群个体数或维度小于3，无法构成向量，直接返回0
    if n < 3 or d < 3:
        return 0.0

    diversity_values = np.zeros(n_samples)

    for sample_idx in range(n_samples):
        # 1. 随机选择三个不同的个体索引
        indiv_indices = np.random.choice(n, 3, replace=False)
        # 2. 随机选择三个不同的维度索引
        dim_indices = np.random.choice(d, 3, replace=False)

        # 3. 提取3x3的矩阵（三个个体在三个维度上的值）
        vectors = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                vectors[i, j] = population[indiv_indices[i], dim_indices[j]]

        # 4. 计算行列式 fz
        # 对于3x3矩阵，手动计算行列式比调用np.linalg.det在Numba中更快/更兼容
        fz = (
                vectors[0, 0] * (vectors[1, 1] * vectors[2, 2] - vectors[1, 2] * vectors[2, 1]) -
                vectors[0, 1] * (vectors[1, 0] * vectors[2, 2] - vectors[1, 2] * vectors[2, 0]) +
                vectors[0, 2] * (vectors[1, 0] * vectors[2, 1] - vectors[1, 1] * vectors[2, 0])
        )

        # 5. 计算向量模长的乘积 fm
        norms = np.zeros(3)
        for i in range(3):
            norm_sq = 0.0
            for j in range(3):
                norm_sq += vectors[i, j] ** 2
            norms[i] = np.sqrt(norm_sq)
        fm = norms[0] * norms[1] * norms[2]

        # 6. 计算并存储本次采样的多样性值
        if fm > epsilon:
            # 取绝对值，因为方向不重要；值域通常在[0,1]
            diversity_values[sample_idx] = np.abs(fz) / fm
        else:
            # 如果fm为0或极小，说明向量太短，本次采样多样性记为0
            diversity_values[sample_idx] = 0.0

    # 7. 返回中位数（比均值对异常值更鲁棒）
    return np.median(diversity_values)
