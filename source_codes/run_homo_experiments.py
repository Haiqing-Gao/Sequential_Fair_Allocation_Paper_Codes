import numpy as np
import pandas as pd
import scipy

from math import gcd
from functools import reduce
import csv
from itertools import product
import ast

def cal_gcd(arr):
    return reduce(gcd, arr)

def count_decimals(number):
    s = str(number)
    if '.' in s:
        return len(s.split('.')[1].rstrip('0'))
    return 0


from Solve_Forward_DP import *
from Solve_Ex_Post_DP import *
from evaluations import *

# Create a list of distributions
D1 = {1: 1/5, 2: 1/5, 3: 1/5, 4: 1/5, 5: 1/5}
D2 = {1: 1/10, 2: 1/5, 3: 2/5, 4: 1/5, 5: 1/10}
D3 = {1: 2/5, 2: 3/10, 3: 1/5, 4: 7/100, 5: 3/100}
D4 = {1: 3/100, 2: 7/100, 3: 1/5, 4: 3/10, 5: 2/5}
D5 = {1: 2/5, 2: 3/40, 3: 1/20, 4: 3/40, 5: 2/5}
D6 = {1: 3/10, 2: 2/5, 3: 1/5, 4: 7/100, 5: 3/100}
D7 = {1: 1/5, 2: 7/100, 3: 3/100, 4: 3/10, 5: 2/5}
D8 = {1: 2/5, 2: 3/100, 3: 3/10, 4: 7/100, 5: 1/5}
# D9 = {10: 1/5, 20: 1/5, 30: 1/5, 40: 1/5, 50: 1/5}
# D10 = {10: 1/10, 20: 1/5, 30: 2/5, 40: 1/5, 50: 1/10}
# D11 = {10: 2/5, 20: 3/10, 30: 1/5, 40: 7/100, 50: 3/100}
# D12 = {10: 3/100, 20: 7/100, 30: 1/5, 40: 3/10, 50: 2/5}
# D13 = {10: 2/5, 20: 3/40, 30: 1/20, 40: 3/40, 50: 2/5}
# D14 = {10: 3/10, 20: 2/5, 30: 1/5, 40: 7/100, 50: 3/100}
# D15 = {10: 1/5, 20: 7/100, 30: 3/100, 40: 3/10, 50: 2/5}
# D16 = {10: 2/5, 20: 3/100, 30: 3/10, 40: 7/100, 50: 1/5}
F = [D1, D2, D3, D4, D5, D6, D7, D8]
# F = [D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16]
# F = [D3]
# Algorithms
# algorithms = ["haiqing", "manshadi", "ppa", "heuristic"]
algorithms = ["haiqing", "manshadi"]
# Distance = [0, 5, 10]
# Distance = [0]
Horizon = [3, 4, 5, 6]
for horizon in Horizon:
    for i in range(len(F)):
        # print(i)
        # store results in a dictionary
        results = {
            "algorithm": [],
            "discretized_distribution": [],
            "horizon": [],
            "supply_divided_by_mean_demand": [],
            "initial_supply": [],
            "haiqing_obj": [],
            "one_time_fairness": [],
            "long_time_fairness": [],
            "ex_ante_preference": [],
            "ex_post_preference": [],
            "ex_ante_envy": [],
            "ex_post_envy": [],
            "expected_waste": [],
            "allocations": [],
            "sample_paths": [],
        }
        for algorithm in algorithms:
            # print(F[i])
            # D = {key + distance: value for key, value in F[i].items()}
            D = F[i]
            demands = list(D.keys())
            d_max = max(demands)
            ### not the true distirbution mean
            d_mean = np.mean(demands)
            # horizon = 3
            # horizon = 6
            auto_step_size = 0.1
            beta_step_size = 0.01
            beta_grid = np.round(np.arange(0, 1 + 1e-10, beta_step_size), count_decimals(beta_step_size))
            # step_size = 0.1
            # auto_step_size = min(step_size, cal_gcd(demands))
            capacity_list = np.arange(0.1, (d_max/d_mean) + 0.1, 0.1)
            c_max = np.max(capacity_list)*d_mean*horizon
            capacity_grid = np.round(np.arange(0, c_max + 1e-10, auto_step_size), count_decimals(auto_step_size))
            if algorithm == "haiqing":
                try:
                    dp_table = {}
                    alloc_table = {}
                    with open(f'./haiqing_dp_table/new_table/brute_force_homo_{horizon}_discrete_type_{i}.csv', mode='r', newline='') as file:
                    # with open(f'./haiqing_dp_table/discrete_type_3.csv', mode='r', newline='') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            state = ast.literal_eval(row["state"])
                            dp_table[state] = float(row["value"])
                            alloc_table[state] = float(row["allocation"])
                except FileNotFoundError:
                    dp_results = {"state": [], "value": [], "allocation": []}
                    dp_table, alloc_table = brute_force_solve_homogeneous_haiqing_DP(D, horizon, capacity_grid, auto_step_size)
                    for state in dp_table.keys():
                        dp_results["state"].append(state)
                        dp_results["value"].append(dp_table[state])
                        dp_results["allocation"].append(alloc_table[state])
                    df = pd.DataFrame(dp_results)
                    df.to_csv(f'./haiqing_dp_table/new_table/brute_force_homo_{horizon}_discrete_type_{i}.csv', index=False)
                # dp_table, alloc_table = solve_homogeneous_haiqing_DP(D, horizon, capacity_grid, auto_step_size)
            elif algorithm == "manshadi":
                try:
                    dp_table = {}
                    alloc_table = {}
                    # with open(f'./manshadi_dp_table/discrete_type_{i}_shift_{distance}.csv', mode='r', newline='') as file:
                    with open(f'./manshadi_dp_table/new_table/brute_force_homo_{horizon}_discrete_type_{i}.csv', mode='r', newline='') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            state = ast.literal_eval(row["state"])
                            dp_table[state] = float(row["value"])
                            alloc_table[state] = float(row["allocation"])
                except FileNotFoundError:
                    dp_results = {"state": [], "value": [], "allocation": []}
                    dp_table, alloc_table = revised_end_solve_homogeneous_manshadi_DP(D, horizon, beta_grid, capacity_grid, auto_step_size)
                    for state in dp_table.keys():
                        dp_results["state"].append(state)
                        dp_results["value"].append(dp_table[state])
                        dp_results["allocation"].append(alloc_table[state])
                    df = pd.DataFrame(dp_results)
                    df.to_csv(f'./manshadi_dp_table/new_table/brute_force_homo_{horizon}_discrete_type_{i}.csv', index=False)
                    # df.to_csv(f'./manshadi_dp_table/discrete_type_{i}_shift_{distance}.csv', index=False)
                # dp_table, alloc_table = solve_homogeneous_manshadi_DP(D, horizon, beta_grid, capacity_grid, auto_step_size)
            else:
                print("Wrong algorithm input. Please check.")
                break
            for capacity_level in capacity_list:
                c1 = round_to_grid(capacity_level*d_mean*horizon, capacity_grid)
                # Get sample-path wise allocations. Prepare for evaluations on metrics
                # # of sample paths
                # n = len(demands)*horizon
                x = []
                d = []
                prob = []
                # calculate Z0
                # obj = 0
                # xx = []
                # dd = []
                # probb = []
                for path in product(D.keys(), repeat=horizon):
                    t = 1
                    c = round_to_grid(c1, capacity_grid)
                    beta = 1
                    p = 1
                    x_path = []
                    d_path = []
                    for demand in path:
                        d_path.append(demand)
                        p *= D[demand]
                        if algorithm == "haiqing":
                            alloc = alloc_table[(t, c, demand)]
                        elif algorithm == "manshadi":
                            alloc = alloc_table[(t, beta, c, demand)]
                        else:
                            print("Wrong algorithm input. Please check.")
                            break
                        x_path.append(alloc)
                        t += 1
                        c = round_to_grid(c - alloc, capacity_grid)
                        beta = round_to_grid(min(beta, alloc/demand), beta_grid)
                    x.append(x_path)
                    d.append(d_path)
                    prob.append(p)
                haiqing_obj = calculate_haiqing_time_zero_objective(c1, D, dp_table, algorithm)
                ex_post_min_fr = calculate_ex_post(det_min_fill_rate, x, d, prob)
                ex_post_preference = calculate_ex_post(det_preference, x, d, prob)
                ex_post_envy = calculate_ex_post(det_envy, x, d, prob)
                expected_min_fr, ex_ante_preference = ex_ante_fr(x, d, prob)
                ex_ante_envy = calculate_ex_ante_envy(x, d, prob)
                expected_waste = calculate_expected_waste(c1, x, prob)
                results["algorithm"].append(algorithm)
                results["discretized_distribution"].append(D)
                results["horizon"].append(horizon)
                results["supply_divided_by_mean_demand"].append(capacity_level)
                results["initial_supply"].append(c1)
                results["haiqing_obj"].append(haiqing_obj)
                results["one_time_fairness"].append(ex_post_min_fr)
                results["long_time_fairness"].append(expected_min_fr)
                results["ex_ante_preference"].append(ex_ante_preference)
                results["ex_post_preference"].append(ex_post_preference)
                results["ex_ante_envy"].append(ex_ante_envy)
                results["ex_post_envy"].append(ex_post_envy)
                results["expected_waste"].append(expected_waste)
                results["allocations"].append(x)
                results["sample_paths"].append(d)
        # print(f'x is {x}, \n d is {d}, \n prob is {prob}')
        # save results
        df = pd.DataFrame(results)
        # df.to_csv(f"./homo_results/discrete_type_{i}_shift_{distance}.csv", index=False)
        df.to_csv(f"./homo_results/brute_force_homo_{horizon}_discrete_type_{i}.csv", index=False)
        # df.to_csv(f"./homo_results/discrete_type_{i}_shift_{distance}.csv", index=False)
