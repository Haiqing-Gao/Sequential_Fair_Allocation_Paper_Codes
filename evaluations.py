import numpy as np
# Ex-post Metrics
#calculate sample-path wise metrics
def det_envy(x, d):
    '''
    :param x: a numpy array of size (horizon)
    :param d:
    :return:
    '''
    N = len(x)
    envy_values = [
        max(min(x[j] / d[i], 1) - min(x[i] / d[i], 1), 0)
        for i in range(N)
        for j in range(N)
    ]
    return max(envy_values)

def det_preference(x, d):
    N = len(x)
    # compare betas
    pref_values = [
        abs(min(x[j] / d[j], 1) - min(x[i] / d[i], 1))
        for i in range(N)
        for j in range(i + 1, N)
    ]
    return max(pref_values)

def det_min_fill_rate(x, d):
    return min([min(x[i] / d[i], 1) for i in range(len(x))])

def calculate_ex_post(metric, x, d, prob):
    value = 0
    for i in range(len(x)):
        value += prob[i] * metric(x[i], d[i])
    return value

# print(f'ex-post envy is {calculate_ex_post(det_envy, x, d, prob)}')
# print(f'ex-post preference is {calculate_ex_post(det_preference, x, d, prob)}')
# print(f'ex-post min fill rate is {calculate_ex_post(det_min_fill_rate, x, d, prob)}')

# Ex-ante type metrics: ex-ante envy, ex-ante preference, expected min fr
def calculate_ex_ante_envy(x, d, prob):
    '''
    :param x: a list of np arrays of dimension horizon
    :param d:
    :return:
    '''
    N = len(x)
    horizon = len(x[0])
    fill_rates = np.zeros((N, horizon, horizon))
    # for every sample path, we calculate the envy of agent i to agent j
    for n in range(N):
        for i in range(horizon):
            for j in range(horizon):
                fill_rates[n, i, j] = prob[n]* min(x[n][j] / d[n][i], 1)
    print(fill_rates)
    expected_envy = fill_rates.sum(axis=0)
    # print(f'expected_envy is {expected_envy}')
    max_envy = [max(np.max(expected_envy[i, :]) - expected_envy[i, i], 0) for i in range(horizon)]
    # print(f'max_envy is {max_envy}')
    return max(max_envy)

def ex_ante_fr(x, d, prob):
    N = len(x)
    # compare betas
    fill_rates = []
    for i in range(N):
        fill_rates.append([prob[i]*min(xi / di, 1) for xi, di in zip(x[i], d[i])])
    # print(f'fill_rates: {fill_rates}')
    expected_fill_rates = np.sum(fill_rates, axis=0)
    # print(f'expected_fill_rates is {expected_fill_rates}')
    expected_min_fill_rate = np.min(expected_fill_rates)
    ex_ante_preference = np.max(expected_fill_rates) - expected_min_fill_rate
    return expected_min_fill_rate, ex_ante_preference

# Efficiency
def calculate_expected_waste(c, x, prob):
    N = len(x)
    expected_efficiency = 0
    for i in range(N):
        expected_efficiency += prob[i]*sum(x[i])
    return c - expected_efficiency

# Evaluate Objective
def calculate_haiqing_time_zero_objective(c, D, dp_table, algorithm):
    Z0 = 0
    for d in D.keys():
        if algorithm == "haiqing":
            Z0 += D[d]*dp_table[(1, c, d)]
        elif algorithm == "manshadi":
            Z0 += D[d]*dp_table[(1, 1, c, d)]
    return Z0

def calculate_dynamic_haiqing_time_zero_objective(c, i0, D, dp_table, algorithm, horizon):
    Z0 = 0
    for d in D.keys():
        if algorithm == "haiqing":
            Z0 += D[d]*dp_table[(1, c, d, i0, tuple(j for j in range(horizon) if j != i0))]
        elif algorithm == "manshadi":
            Z0 += D[d]*dp_table[(1, 1, c, d, i0, tuple(j for j in range(horizon) if j != i0))]
    return Z0

# def calculate_haiqing_time_zero_objective(x, d, dist):
#     Z0 = 0
#     horizon = len(x[0])
#     for t in range(horizon, 0, -1):
#         if t == horizon:
#
#     Z0 = 0
#     for d in D.keys():
#         Z0 += D[d]*dp_table[(1, c, d)]
#     return Z0

# print(a)