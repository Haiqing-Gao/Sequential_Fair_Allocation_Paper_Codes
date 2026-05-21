import numpy as np
import pandas as pd
import scipy
import math
import os
import csv
import itertools
import random
from itertools import product
import ast
from math import gcd
from functools import reduce

from Solve_Forward_DP import *
from Solve_Ex_Post_DP import *
from evaluations import *

def cal_gcd(arr):
    return reduce(gcd, arr)

def count_decimals(number):
    s = str(number)
    if '.' in s:
        return len(s.split('.')[1].rstrip('0'))
    return 0

def calculate_CV(prob_dict):
    mean = sum(x * p for x, p in prob_dict.items())
    variance = sum(p * (x - mean) ** 2 for x, p in prob_dict.items())
    std = math.sqrt(variance)
    return std, std / mean

def calculate_mean_variance(prob_dict):
    mean = sum(x * p for x, p in prob_dict.items())
    variance = sum(p * (x - mean) ** 2 for x, p in prob_dict.items())
    return mean, variance

def is_decreasing_CV(distr_list, allow_equal=True, tol=1e-12):
    """Return True if list of distributions is sorted by decreasing CV."""
    cvs = []
    for distr in distr_list:
        _, cv = calculate_CV(distr)
        cvs.append(cv)
    for left, right in zip(cvs, cvs[1:]):
        if allow_equal:
            if left + tol < right:
                return False
        else:
            if left <= right + tol:
                return False
    return True

def is_decreasing_mean(distr_list, allow_equal=True, tol=1e-12):
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

def is_decreasing_variance(distr_list, allow_equal=True, tol=1e-12):
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

def get_dp_table_path(algorithm, setting, route, type_index, shift_or_mean_or_cv):
    if setting == "same_var":
        subdir = "hete_static_same_var"
        filename = f"hete_same_var_type_{type_index}_shift_{shift_or_mean_or_cv}_{route}.csv"
    elif setting == "same_mean":
        subdir = "hete_static_same_mean"
        filename = f"hete_same_mean_type_{type_index}_mean_{shift_or_mean_or_cv}_{route}.csv"
    elif setting == "same_CV":
        subdir = "hete_static_same_CV"
        filename = f"hete_same_CV_type_{type_index}_CV_{shift_or_mean_or_cv}_{route}.csv"
    elif setting == "random":
        subdir = "hete_static_random"
        filename = f"type_{type_index}_random_horizon_{horizon}_{route}.csv"
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


def get_dp_table_path_precision_version(algorithm, setting, route, type_index, shift_or_mean_or_cv):
    subdir = "testing_precision_real_dynamic"
    if setting == "same_var":
        filename = f"hete_same_var_type_{type_index}_shift_{shift_or_mean_or_cv}_{route}.csv"
    elif setting == "same_mean":
        filename = f"hete_same_mean_type_{type_index}_mean_{shift_or_mean_or_cv}_{route}.csv"
    elif setting == "same_CV":
        filename = f"hete_same_CV_type_{type_index}_CV_{shift_or_mean_or_cv}_{route}.csv"
    elif setting == "random":
        filename = f"hete_random_type_{type_index}_horizon_{horizon}_{route}.csv"
    else:
        raise ValueError(f"Unknown setting: {setting}")

    if algorithm == "haiqing":
        path = os.path.join(subdir, "./haiqing_dp_table")
    elif algorithm == "manshadi":
        path = os.path.join(subdir, "./manshadi_dp_table")
    else:
        return None

    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)

def save_dp_table(path, dp_table, alloc_table):
    dp_results = {"state": [], "value": [], "allocation": []}
    for state in dp_table.keys():
        dp_results["state"].append(state)
        dp_results["value"].append(dp_table[state])
        dp_results["allocation"].append(alloc_table[state])
    df = pd.DataFrame(dp_results)
    df.to_csv(path, index=False)

def load_dp_table(path):
    dp_table = {}
    alloc_table = {}
    with open(path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            state = ast.literal_eval(row["state"])
            dp_table[state] = float(row["value"])
            alloc_table[state] = float(row["allocation"])
    return dp_table, alloc_table

# Base distributions
D1 = {1: 1/5, 2: 1/5, 3: 1/5, 4: 1/5, 5: 1/5}
D2 = {1: 1/10, 2: 1/5, 3: 2/5, 4: 1/5, 5: 1/10}
D3 = {1: 2/5, 2: 3/10, 3: 1/5, 4: 7/100, 5: 3/100}
D4 = {1: 3/100, 2: 7/100, 3: 1/5, 4: 3/10, 5: 2/5}
D5 = {1: 2/5, 2: 3/40, 3: 1/20, 4: 3/40, 5: 2/5}
D6 = {1: 3/10, 2: 2/5, 3: 1/5, 4: 7/100, 5: 3/100}
D7 = {1: 1/5, 2: 7/100, 3: 3/100, 4: 3/10, 5: 2/5}
D8 = {1: 2/5, 2: 3/100, 3: 3/10, 4: 7/100, 5: 1/5}
# distr_list = [D1, D2, D3, D4, D5, D6, D7, D8]
distr_list = [D1, D2, D3, D7, D8]

horizon = 3
algorithms = ["haiqing", "manshadi"]
settings = ["same_var", "same_mean", "same_CV", "random"]

for setting in settings:
    print(f"Starting setting: {setting}")
    if setting == "same_var":
        shifts = [0.5, 1.0]
        type_range = range(len(distr_list))
    elif setting == "same_mean":
        mean_val = 3.5
        shifts = [mean_val]
        new_list = []
        for distr in distr_list:
            distr_mean = sum(value * prob for value, prob in distr.items())
            shift = round(mean_val - distr_mean, 1)
            D = {round(key + shift, 1): prob for key, prob in distr.items()}
            new_list.append(D)
        combinations = list(itertools.combinations(new_list, horizon))
        type_range = range(len(combinations))
    elif setting == "same_CV":
        cv_val = 0.35140452079103174
        shifts = [cv_val]
        new_list = []
        for distr in distr_list:
            distr_mean = sum(x * p for x, p in distr.items())
            std, _ = calculate_CV(distr)
            shift = round(std / cv_val - distr_mean, 1)
            D = {round(key + shift, 1): prob for key, prob in distr.items()}
            new_list.append(D)
        combinations = list(itertools.combinations(new_list, horizon))
        type_range = range(len(combinations))
    elif setting == "random":
        shifts = [0]
        # Use all combinations of horizon distributions from distr_list
        combinations = list(itertools.combinations(distr_list, horizon))
        type_range = range(len(combinations))

    for shift in shifts:
        print(f"  Shift/Mean/CV value: {shift}")
        for i in type_range:
            res_dir = f"./testing_precision_real_dynamic/results"
            os.makedirs(res_dir, exist_ok=True)
            if setting == "random":
                suffix = f"{setting}_type_{i}_horizon_{horizon}.csv"
            else:
                suffix = f"{setting}_type_{i}_shift_{shift}.csv"
            
            # Check if results already exist for this type and shift
            result_files = [
                os.path.join(res_dir, f"best_static_{suffix}"),
                os.path.join(res_dir, f"worst_static_{suffix}"),
                os.path.join(res_dir, f"decreasing_CV_{suffix}"),
                os.path.join(res_dir, f"increasing_CV_{suffix}")
            ]
            if all(os.path.exists(f) for f in result_files):
                print(f"    Skipping type {i} for {setting} shift {shift} as results already exist.")
                continue

            if setting == "same_var":
                F = []
                D_base = distr_list[i]
                for n in range(horizon):
                    D = {round(key + n * shift, 1): prob for key, prob in D_base.items()}
                    F.append(D)
            else:
                F = list(combinations[i])

            d_max = sum(max(node.keys()) for node in F)
            d_mean = sum(sum(value * prob for value, prob in node.items()) for node in F)
            routes = list(itertools.permutations(F))
            
            best_static_results = {k: [] for k in ["algorithm", "discretized_distribution", "horizon", "supply_divided_by_mean_demand", "initial_supply", "haiqing_obj", "one_time_fairness", "long_time_fairness", "ex_ante_preference", "ex_post_preference", "ex_ante_envy", "ex_post_envy", "expected_waste", "allocations", "sample_paths"]}
            worst_static_results = {k: [] for k in ["algorithm", "discretized_distribution", "horizon", "supply_divided_by_mean_demand", "initial_supply", "haiqing_obj", "one_time_fairness", "long_time_fairness", "ex_ante_preference", "ex_post_preference", "ex_ante_envy", "ex_post_envy", "expected_waste", "allocations", "sample_paths"]}
            decreasing_CV_results = {k: [] for k in ["algorithm", "discretized_distribution", "horizon", "supply_divided_by_mean_demand", "initial_supply", "haiqing_obj", "one_time_fairness", "long_time_fairness", "ex_ante_preference", "ex_post_preference", "ex_ante_envy", "ex_post_envy", "expected_waste", "allocations", "sample_paths"]}
            increasing_CV_results = {k: [] for k in ["algorithm", "discretized_distribution", "horizon", "supply_divided_by_mean_demand", "initial_supply", "haiqing_obj", "one_time_fairness", "long_time_fairness", "ex_ante_preference", "ex_post_preference", "ex_ante_envy", "ex_post_envy", "expected_waste", "allocations", "sample_paths"]}

            auto_step_size = 0.05
            # auto_step_size = 0.1
            beta_step_size = 0.01
            beta_grid = np.round(np.arange(0, 1 + 1e-10, beta_step_size), count_decimals(beta_step_size))
            capacity_list = np.arange(0.1, (d_max / d_mean) + 0.1, 0.1)
            c_max = np.max(capacity_list) * d_mean
            capacity_grid = np.round(np.arange(0, c_max + 1e-10, auto_step_size), count_decimals(auto_step_size))

            for algorithm in algorithms:
                for capacity_level in capacity_list:
                    c1 = round_to_grid(capacity_level * d_mean, capacity_grid)
                    routes_dict = {}
                    best_obj = -1e9
                    worst_obj = 2
                    
                    for route in routes:
                        if setting == "same_CV":
                            # Check "mean" or "variance" when setting is "same_CV"
                            # Using mean by default as per previous logic in hete_calculate_mertics.py
                            if is_decreasing_mean(route):
                                routes_dict["decreasing_CV"] = route
                                routes_dict["increasing_CV"] = route[::-1]
                                break
                            # If checking variance was requested:
                            # if is_decreasing_variance(route):
                            #     routes_dict["decreasing_CV"] = route
                            #     routes_dict["increasing_CV"] = route[::-1]
                            #     break
                        else:
                            if is_decreasing_CV(route):
                                routes_dict["decreasing_CV"] = route
                                routes_dict["increasing_CV"] = route[::-1]
                                break
                    
                    for route in routes:
                        # dp_path = get_dp_table_path(algorithm, setting, route, i, shift)
                        dp_path = get_dp_table_path_precision_version(algorithm, setting, route, i, shift)
                        if os.path.exists(dp_path):
                            dp_table, _ = load_dp_table(dp_path)
                        else:
                            if algorithm == "haiqing":
                                dp_table, alloc_table = base_solve_heterogeneous_haiqing_DP(route, horizon, capacity_grid, auto_step_size)
                            else: # manshadi
                                dp_table, alloc_table = base_heterogeneous_manshadi_DP(route, horizon, beta_grid, capacity_grid, auto_step_size)
                            save_dp_table(dp_path, dp_table, alloc_table)
                        
                        if algorithm == "haiqing":
                            current_obj = calculate_haiqing_time_zero_objective(c1, route[0], dp_table, algorithm)
                        else:
                            current_obj = calculate_manshadi_time_zero_objective(c1, route[0], dp_table)
                        
                        if current_obj > best_obj:
                            routes_dict["best_route"] = route
                            best_obj = current_obj
                        if current_obj < worst_obj:
                            routes_dict["worst_route"] = route
                            worst_obj = current_obj

                    for r_key in ["best_route", "worst_route", "decreasing_CV", "increasing_CV"]:
                        route = routes_dict[r_key]
                        # dp_path = get_dp_table_path_(algorithm, setting, route, i, shift)
                        dp_path = get_dp_table_path_precision_version(algorithm, setting, route, i, shift)
                        if os.path.exists(dp_path):
                            dp_table, alloc_table = load_dp_table(dp_path)
                        else:
                            if algorithm == "haiqing":
                                dp_table, alloc_table = base_solve_heterogeneous_haiqing_DP(route, horizon, capacity_grid, auto_step_size)
                            else: # manshadi
                                dp_table, alloc_table = base_heterogeneous_manshadi_DP(route, horizon, beta_grid, capacity_grid, auto_step_size)
                            save_dp_table(dp_path, dp_table, alloc_table)
                        
                        x, d, prob = [], [], []
                        for path in product(*route):
                            t, c, beta, p = 1, c1, 1, 1
                            x_path, d_path = [], []
                            for demand in path:
                                d_path.append(demand)
                                p *= route[t-1][demand]
                                if algorithm == "haiqing":
                                    alloc = alloc_table.get((t, round_to_grid(c, capacity_grid), demand))
                                    if alloc is None:
                                        # Fallback or error reporting
                                        print(f"Warning: Key ({(t, round_to_grid(c, capacity_grid), demand)}) not found in alloc_table. Capacity grid size: {len(capacity_grid)}")
                                        alloc = min(c, demand) # Heuristic fallback
                                else: # manshadi
                                    alloc = alloc_table.get((t, round_to_grid(beta, beta_grid), round_to_grid(c, capacity_grid), demand))
                                    if alloc is None:
                                        print(f"Warning: Key ({(t, round_to_grid(beta, beta_grid), round_to_grid(c, capacity_grid), demand)}) not found in alloc_table.")
                                        alloc = min(c, demand) # Heuristic fallback
                                x_path.append(alloc)
                                c = round_to_grid(c - alloc, capacity_grid)
                                beta = round_to_grid(min(beta, alloc / demand), beta_grid)
                                t += 1
                            x.append(x_path)
                            d.append(d_path)
                            prob.append(p)

                        haiqing_obj = calculate_haiqing_time_zero_objective(c1, route[0], dp_table, algorithm) if algorithm == "haiqing" else calculate_manshadi_time_zero_objective(c1, route[0], dp_table)
                        ex_post_min_fr = calculate_ex_post(det_min_fill_rate, x, d, prob)
                        ex_post_preference = calculate_ex_post(det_preference, x, d, prob)
                        ex_post_envy = calculate_ex_post(det_envy, x, d, prob)
                        expected_min_fr, ex_ante_preference = ex_ante_fr(x, d, prob)
                        ex_ante_envy = calculate_ex_ante_envy(x, d, prob)
                        expected_waste = calculate_expected_waste(c1, x, prob)

                        if r_key == "best_route":
                            target_results = best_static_results
                        elif r_key == "worst_route":
                            target_results = worst_static_results
                        elif r_key == "decreasing_CV":
                            target_results = decreasing_CV_results
                        else: # increasing_CV
                            target_results = increasing_CV_results
                        
                        target_results["algorithm"].append(algorithm)
                        target_results["discretized_distribution"].append(route)
                        target_results["horizon"].append(horizon)
                        target_results["supply_divided_by_mean_demand"].append(capacity_level)
                        target_results["initial_supply"].append(c1)
                        target_results["haiqing_obj"].append(haiqing_obj)
                        target_results["one_time_fairness"].append(ex_post_min_fr)
                        target_results["long_time_fairness"].append(expected_min_fr)
                        target_results["ex_ante_preference"].append(ex_ante_preference)
                        target_results["ex_post_preference"].append(ex_post_preference)
                        target_results["ex_ante_envy"].append(ex_ante_envy)
                        target_results["ex_post_envy"].append(ex_post_envy)
                        target_results["expected_waste"].append(expected_waste)
                        target_results["allocations"].append(x)
                        target_results["sample_paths"].append(d)

            # res_dir = f"./hete_static_{setting}_results"
            # os.makedirs(res_dir, exist_ok=True)
            # if setting == "same_var":
            #     suffix = f"type_{i}_shift_{shift}.csv"
            # elif setting == "same_mean":
            #     suffix = f"type_{i}_mean_{shift}.csv"
            # else: # same_CV
            #     suffix = f"type_{i}_CV_{shift}.csv"
            pd.DataFrame(best_static_results).to_csv(os.path.join(res_dir, f"best_static_{suffix}"), index=False)
            pd.DataFrame(worst_static_results).to_csv(os.path.join(res_dir, f"worst_static_{suffix}"), index=False)
            pd.DataFrame(decreasing_CV_results).to_csv(os.path.join(res_dir, f"decreasing_CV_{suffix}"), index=False)
            pd.DataFrame(increasing_CV_results).to_csv(os.path.join(res_dir, f"increasing_CV_{suffix}"), index=False)
