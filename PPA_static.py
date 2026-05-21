import numpy as np
import pandas as pd
import scipy

from math import gcd
from functools import reduce
import csv
import itertools
from itertools import product
import ast
import random
import os


def cal_gcd(arr):
    return reduce(gcd, arr)

def count_decimals(number):
    s = str(number)
    if '.' in s:
        return len(s.split('.')[1].rstrip('0'))
    return 0


def calculate_CV(prob_dict):
    # Calculate the mean
    import math
    mean = sum(x * p for x, p in prob_dict.items())

    # Calculate the variance
    variance = sum(p * (x - mean) ** 2 for x, p in prob_dict.items())
    std = math.sqrt(variance)

    return std, std / mean

def calculate_mean_variance(prob_dict):
    mean = sum(x * p for x, p in prob_dict.items())
    variance = sum(p * (x - mean) ** 2 for x, p in prob_dict.items())
    return mean, variance

def is_decreasing_cv(distr_list, allow_equal=True, tol=1e-3):
    """Return True if list of distributions is sorted by decreasing CV."""
    cvs = []
    for distr in distr_list:
        _, cv = calculate_CV(distr)
        cvs.append(cv)
    for left, right in zip(cvs, cvs[1:]):
        if allow_equal:
            if left + tol < right:
                # left >= right, allow there is a precision error so that left > right + tol
                return False
        else:
            if left <= right + tol:
                # left > right, and should be at least left - right > tol
                return False
    return True

def is_decreasing_mean(distr_list, allow_equal=True, tol=1e-3):
    """Return True if list of distributions is sorted by decreasing mean."""
    means = []
    for distr in distr_list:
        mean, _ = calculate_mean_variance(distr)
        means.append(mean)
    for left, right in zip(means, means[1:]):
        if allow_equal:
            if left + tol < right:
                return False
        else:
            if left <= right + tol:
                return False
    return True

def is_decreasing_variance(distr_list, allow_equal=True, tol=1e-3):
    """Return True if list of distributions is sorted by decreasing variance."""
    variances = []
    for distr in distr_list:
        _, variance = calculate_mean_variance(distr)
        variances.append(variance)
    for left, right in zip(variances, variances[1:]):
        if allow_equal:
            if left + tol < right:
                return False
        else:
            if left <= right + tol:
                return False
    return True

def solve_ppa(joint_dist, horizon, capacity_grid, beta_grid, algorithm):
    # calculate haiqing objective
    dp_table = {}
    # allocation table
    alloc_table = {}
    for t in range(horizon, 0, -1):
        current_demands = list(joint_dist[t-1].keys())
        for c in capacity_grid:
            c = round_to_grid(c, capacity_grid)
            for d in current_demands:
                if t == horizon:
                    if algorithm == "haiqing":
                        dp_table[(t, c, d)] = min(c / d, 1)
                        alloc_table[(t, c, d)] = min(c, d)
                    elif algorithm == "manshadi":
                        for beta in beta_grid:
                            beta = round_to_grid(beta, beta_grid)
                            # print(f'we have state {(t, beta, c ,d)}')
                            dp_table[(t, beta, c, d)] = min(c / d, beta)
                            alloc_table[(t, beta, c, d)] = min(c, d)
                else:
                    future_mean = sum(value * prob  for n in range(t, horizon) for value, prob in joint_dist[n].items())
                    current_beta = min(c/(d + future_mean), 1)
                    if algorithm == "haiqing":
                        alloc_table[(t, c, d)] = min(d*c/(d + future_mean), d)
                        next_c = round_to_grid(c - alloc_table[(t, c, d)], capacity_grid)
                        dp_table[(t, c, d)] = sum(prob*min(current_beta, dp_table[(t+1, next_c, value)]) for value, prob in joint_dist[t].items())
                        # print(f'at stage {t}, future mean is {future_mean} by summing up {range(t, horizon)} and obj is {dp_table[(t, c, d)]} when state is {(t, c, d)}.')
                    elif algorithm == "manshadi":
                        for beta in beta_grid:
                            beta = round_to_grid(beta, beta_grid)
                            alloc_table[(t, beta, c, d)] = min(d*c/(d + future_mean), d)
                            next_c = round_to_grid(c - alloc_table[(t, beta, c, d)], capacity_grid)
                            next_beta = round_to_grid(min(current_beta, beta), beta_grid)
                            dp_table[(t, beta, c, d)] = sum(prob*dp_table[(t+1, next_beta, next_c, value)] for value, prob in joint_dist[t].items())
    return dp_table, alloc_table


def get_ppa_dp_table_path(algorithm, setting, route, type_index, shift):
    # route is the distribution tuple
    if setting == "same_var":
        filename = f"ppa_{route}_type_{type_index}_shift_{shift}.csv"
        subdir = "PPA_same_var"
    elif setting == "same_mean":
        filename = f"ppa_{route}_type_{type_index}_mean_{shift}.csv"
        subdir = "PPA_same_mean"
    elif setting == "same_CV":
        filename = f"ppa_{route}_type_{type_index}_CV_{shift}.csv"
        subdir = "PPA_same_CV"
    elif setting == "random":
        filename = f"type_{type_index}_random_horizon_{horizon}_{route}.csv"
        subdir = "PPA_random"
    else:
        raise ValueError(f"Unknown setting: {setting}")

    if algorithm == "haiqing":
        path = os.path.join("./haiqing_dp_table", subdir)
    elif algorithm == "manshadi":
        path = os.path.join("./manshadi_dp_table", subdir)
    else:
        return None
    
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)


def save_ppa_dp_table(path, dp_table, alloc_table):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["state", "value", "allocation"])
        for state in dp_table.keys():
            writer.writerow([state, dp_table[state], alloc_table[state]])


def load_ppa_dp_table(path):
    dp_table = {}
    alloc_table = {}
    with open(path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            state = ast.literal_eval(row["state"])
            dp_table[state] = float(row["value"])
            alloc_table[state] = float(row["allocation"])
    return dp_table, alloc_table


from Solve_Forward_DP import *
from Solve_Ex_Post_DP import *
from evaluations import *

# if __name__ == '__main__':
    # D1 = {1:0.5, 2:0.5}
    # D2 = {6:0.5, 7:0.5}
    # F = [D1, D2]
    # horizon = 2
    # capacity_grid = np.round(np.arange(0, 10 + 1e-10, 0.01), count_decimals(0.01))
    # dp_table, alloc_table = solve_heterogeneous_haiqing_DP(F, horizon, capacity_grid, 0.01)
# Create a list of distributions
D1 = {1: 1/5, 2: 1/5, 3: 1/5, 4: 1/5, 5: 1/5}
D2 = {1: 1/10, 2: 1/5, 3: 2/5, 4: 1/5, 5: 1/10}
D3 = {1: 2/5, 2: 3/10, 3: 1/5, 4: 7/100, 5: 3/100}
D4 = {1: 3/100, 2: 7/100, 3: 1/5, 4: 3/10, 5: 2/5}
D5 = {1: 2/5, 2: 3/40, 3: 1/20, 4: 3/40, 5: 2/5}
D6 = {1: 3/10, 2: 2/5, 3: 1/5, 4: 7/100, 5: 3/100}
D7 = {1: 1/5, 2: 7/100, 3: 3/100, 4: 3/10, 5: 2/5}
D8 = {1: 2/5, 2: 3/100, 3: 3/10, 4: 7/100, 5: 1/5}
distr_list = [D1, D2, D3, D4, D5, D6, D7, D8]
# Algorithms
# algorithms = ["haiqing", "manshadi", "ppa", "heuristic"]
horizon = 3
# algorithms = ["haiqing_ppa"]
algorithms = ["haiqing_ppa", "manshadi_ppa"]
# settings = ["same_var", "same_mean", "same_CV", "random"]
settings = ["random"]
for setting in settings:
    if setting == "same_var":
        shifts = [0.5, 1.0]
        type_range = range(len(distr_list))
    elif setting == "same_mean":
        mean = 3.5
        shifts = [mean]
        new_list = []
        for distr in distr_list:
            distr_mean = sum(value * prob for value, prob in distr.items())
            shift_val = round(mean - distr_mean, 1)
            D = {}
            for key in distr.keys():
                D[round(key + shift_val, 1)] = distr[key]
            new_list.append(D)
        combinations = list(itertools.combinations(new_list, horizon))
        type_range = range(len(combinations))
    elif setting == "same_CV":
        _, CV = calculate_CV(D1)
        # CV -= 0.12 # This was in hete_calculate_mertics.py
        # Actually in hete_calculate_mertics.py it was:
        # _, CV = calculate_CV(D1)
        # CV -= 0.12
        # shifts = [CV]
        # Let's check hete_calculate_mertics.py again.
        # It seems they used a fixed CV for the whole list.
        CV_val = 0.35140452079103174 # This is what create_metric_figure.py expects for same_CV
        shifts = [CV_val]
        new_list = []
        for distr in distr_list:
            distr_mean = sum(x * p for x, p in distr.items())
            std, _ = calculate_CV(distr)
            shift_val = round(std / CV_val - distr_mean, 1)
            D = {}
            for key in distr.keys():
                D[round(key + shift_val, 1)] = distr[key]
            new_list.append(D)
        combinations = list(itertools.combinations(new_list, horizon))
        type_range = range(len(combinations))
    elif setting == "random":
        shifts = [0]
        # Use all combinations of horizon distributions from distr_list
        combinations = list(itertools.combinations(distr_list, horizon))
        type_range = range(len(combinations))

    for shift in shifts:
        for i in type_range:
            # create nodes
            if setting == "same_var":
                F = []
                D_base = distr_list[i]
                for n in range(horizon):
                    D = {}
                    for key in D_base.keys():
                        D[key + n*shift] = D_base[key]
                    F.append(D)
            else:
                F = list(combinations[i])

            d_max = 0
            d_mean = 0
            for D in F:
                d_max += max(list(D.keys()))
                d_mean += sum(value * prob for value, prob in D.items())
            
            routes = list(itertools.permutations(F))
            best_static_results = {
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
            worst_static_results = {
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
            decreasing_CV_results = {
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
            increasing_CV_results = {
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
            auto_step_size = 0.1
            beta_step_size = 0.01
            beta_grid = np.round(np.arange(0, 1 + 1e-10, beta_step_size), count_decimals(beta_step_size))
            # step_size = 0.1
            # auto_step_size = min(step_size, cal_gcd(demands))
            capacity_list = np.arange(0.1, (d_max / d_mean) + 0.1, 0.1)
            c_max = np.max(capacity_list) * d_mean
            capacity_grid = np.round(np.arange(0, c_max + 1e-10, auto_step_size), count_decimals(auto_step_size))
            for algorithm in algorithms:
                for capacity_level in capacity_list:
                    c1 = round_to_grid(capacity_level * d_mean, capacity_grid)
                    routes_dict = {}
                    best_obj = 0
                    worst_obj = 1
                    for route in routes:
                        if is_decreasing_cv(route):
                            routes_dict["decreasing_CV"] = route
                            routes_dict["increasing_CV"] = route[::-1]
                            break
                        if algorithm == "haiqing_ppa":
                            dp_path = get_ppa_dp_table_path("haiqing", setting, route, i, shift)
                            if os.path.exists(dp_path):
                                dp_table, alloc_table = load_ppa_dp_table(dp_path)
                            else:
                                dp_table, alloc_table = solve_ppa(route, horizon, capacity_grid, beta_grid, "haiqing")
                                save_ppa_dp_table(dp_path, dp_table, alloc_table)
                            current_obj = calculate_haiqing_time_zero_objective(c1, route[0], dp_table, "haiqing")
                            if current_obj > best_obj:
                                routes_dict["best_route"] = route
                                best_obj = current_obj
                            if current_obj < worst_obj:
                                routes_dict["worst_route"] = route
                                worst_obj = current_obj
                        elif algorithm == "manshadi_ppa":
                            dp_path = get_ppa_dp_table_path("manshadi", setting, route, i, shift)
                            if os.path.exists(dp_path):
                                dp_table, alloc_table = load_ppa_dp_table(dp_path)
                            else:
                                dp_table, alloc_table = solve_ppa(route, horizon, capacity_grid, beta_grid, "manshadi")
                                save_ppa_dp_table(dp_path, dp_table, alloc_table)
                            current_obj = calculate_manshadi_time_zero_objective(c1, route[0], dp_table)
                            if current_obj > best_obj:
                                routes_dict["best_route"] = route
                                best_obj = current_obj
                            if current_obj < worst_obj:
                                routes_dict["worst_route"] = route
                                worst_obj = current_obj
                        else:
                            print("Wrong algorithm input. Please check.")
                            break
                    for r in routes_dict.keys():
                        x = []
                        d = []
                        prob = []
                        current_route = routes_dict[r]
                        if algorithm == "haiqing_ppa":
                            dp_path = get_ppa_dp_table_path("haiqing", setting, current_route, i, shift)
                            if os.path.exists(dp_path):
                                dp_table, alloc_table = load_ppa_dp_table(dp_path)
                            else:
                                dp_table, alloc_table = solve_ppa(current_route, horizon, capacity_grid, beta_grid, "haiqing")
                                save_ppa_dp_table(dp_path, dp_table, alloc_table)
                        elif algorithm == "manshadi_ppa":
                            dp_path = get_ppa_dp_table_path("manshadi", setting, current_route, i, shift)
                            if os.path.exists(dp_path):
                                dp_table, alloc_table = load_ppa_dp_table(dp_path)
                            else:
                                dp_table, alloc_table = solve_ppa(current_route, horizon, capacity_grid, beta_grid, "manshadi")
                                save_ppa_dp_table(dp_path, dp_table, alloc_table)

                        for path in product(*current_route):
                            t = 1
                            c = round_to_grid(c1, capacity_grid)
                            beta = 1
                            p = 1
                            x_path = []
                            d_path = []
                            for demand in path:
                                d_path.append(demand)
                                p *= current_route[t-1][demand]
                                if algorithm == "haiqing_ppa":
                                    alloc = alloc_table[(t, c, demand)]
                                elif algorithm == "manshadi_ppa":
                                    alloc = alloc_table[(t, beta, c, demand)]
                                else:
                                    print("Wrong algorithm input. Please check.")
                                    break
                                x_path.append(alloc)
                                t += 1
                                c = round_to_grid(c - alloc, capacity_grid)
                                beta = round_to_grid(min(beta, alloc / demand), beta_grid)
                            x.append(x_path)
                            d.append(d_path)
                            prob.append(p)

                        if algorithm == "haiqing_ppa":
                            haiqing_obj = calculate_haiqing_time_zero_objective(c1, current_route[0], dp_table, "haiqing")
                        elif algorithm == "manshadi_ppa":
                            haiqing_obj = calculate_haiqing_time_zero_objective(c1, current_route[0], dp_table, "manshadi")

                        ex_post_min_fr = calculate_ex_post(det_min_fill_rate, x, d, prob)
                        ex_post_preference = calculate_ex_post(det_preference, x, d, prob)
                        ex_post_envy = calculate_ex_post(det_envy, x, d, prob)
                        expected_min_fr, ex_ante_preference = ex_ante_fr(x, d, prob)
                        ex_ante_envy = calculate_ex_ante_envy(x, d, prob)
                        expected_waste = calculate_expected_waste(c1, x, prob)

                        res_dict = None
                        if r == "best_route":
                            res_dict = best_static_results
                        elif r == "worst_route":
                            res_dict = worst_static_results
                        elif r == "decreasing_CV":
                            res_dict = decreasing_CV_results
                        elif r == "increasing_CV":
                            res_dict = increasing_CV_results

                        if res_dict is not None:
                            res_dict["algorithm"].append(algorithm)
                            res_dict["discretized_distribution"].append(current_route)
                            res_dict["horizon"].append(horizon)
                            res_dict["supply_divided_by_mean_demand"].append(capacity_level)
                            res_dict["initial_supply"].append(c1)
                            res_dict["haiqing_obj"].append(haiqing_obj)
                            res_dict["one_time_fairness"].append(ex_post_min_fr)
                            res_dict["long_time_fairness"].append(expected_min_fr)
                            res_dict["ex_ante_preference"].append(ex_ante_preference)
                            res_dict["ex_post_preference"].append(ex_post_preference)
                            res_dict["ex_ante_envy"].append(ex_ante_envy)
                            res_dict["ex_post_envy"].append(ex_post_envy)
                            res_dict["expected_waste"].append(expected_waste)
                            res_dict["allocations"].append(x)
                            res_dict["sample_paths"].append(d)
                            df = pd.DataFrame(res_dict)

                            if setting == "same_var":
                                res_dir = "./ppa_same_var_results"
                                res_file = f"ppa_{r}_type_{i}_shift_{shift}.csv"
                            elif setting == "same_mean":
                                res_dir = "./ppa_same_mean_results"
                                res_file = f"ppa_{r}_type_{i}_mean_{shift}.csv"
                            elif setting == "random":
                                res_dir = "./ppa_random_results"
                                res_file = f"ppa_{r}_type_{i}_random_horizon_{horizon}.csv"
                            
                            os.makedirs(res_dir, exist_ok=True)
                            df.to_csv(os.path.join(res_dir, res_file), index=False)

        # df_best = pd.DataFrame(best_static_results)
        # # df_best.to_csv(f"./hete_results/{algorithm}_best_static_type_{n}.csv", index=False)
        # df_best.to_csv(f"./same_var_results/ppa_best_static_type_{i}_shift_{shift}.csv", index=False)
        # df_worst = pd.DataFrame(worst_static_results)
        # # df_worst.to_csv(f"./hete_results/{algorithm}_worst_static_type_{n}.csv", index=False)
        # df_worst.to_csv(f"./same_var_results/ppa_worst_static_type_{i}_shift_{shift}.csv", index=False)
        #
        # df_decreasing = pd.DataFrame(decreasing_CV_results)
        # df_decreasing.to_csv(f"./same_var_results/ppa_decreasing_CV_type_{i}_shift_{shift}.csv", index=False)
        # df_increasing = pd.DataFrame(increasing_CV_results)
        # df_increasing.to_csv(f"./same_var_results/ppa_increasing_CV_type_{i}_shift_{shift}.csv", index=False)

