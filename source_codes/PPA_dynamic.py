from itertools import product
from Solve_Ex_Post_DP import *
from Solve_Forward_DP import solve_dynamic_haiqing_DP
# count and round functions
from evaluations import *
from generate_distribution import *
from run_dynamic_combined import choose_first_node, dfs
import os
import itertools
import random

def solve_dynamic_ppa(joint_dist, horizon, capacity_grid, beta_grid, algorithm):
    # calculate haiqing objective
    dp_table = {}
    # allocation table
    alloc_table = {}
    # routing table
    route_table = {}
    
    # Calculate means for all distributions in joint_dist
    dist_means = [sum(v * p for v, p in d.items()) for d in joint_dist]

    for t in range(horizon, 0, -1):
        # S is the set of ALREADY visited nodes
        for S in list(itertools.combinations(range(horizon), horizon - t)):
            pool = list(set(range(horizon)) - set(S))
            for i in pool:
                current_demands = list(joint_dist[i].keys())
                # Remaining unvisited nodes AFTER visiting node i
                # remaining_unvisited = set(pool) - {i}
                # future_mean = sum(dist_means[k] for k in remaining_unvisited)
                future_mean = sum(dist_means[k] for k in set(S))

                for c in capacity_grid:
                    c = round_to_grid(c, capacity_grid)
                    for d in current_demands:
                        current_beta = min(c / (d + future_mean), 1)
                        if t == horizon:
                            if algorithm == "haiqing_ppa":
                                dp_table[(t, c, d, i, S)] = min(c / d, 1)
                                alloc_table[(t, c, d, i, S)] = min(c, d)
                                route_table[(t, c, d, i, S)] = ()
                            elif algorithm == "manshadi_ppa":
                                for beta in beta_grid:
                                    beta = round_to_grid(beta, beta_grid)
                                    dp_table[(t, beta, c, d, i, S)] = min(c / d, beta)
                                    alloc_table[(t, beta, c, d, i, S)] = min(c, d)
                                    route_table[(t, beta, c, d, i, S)] = ()
                        else:
                            if algorithm == "haiqing_ppa":
                                alloc = min(d * c / (d + future_mean), d)
                                next_c = round_to_grid(c - alloc, capacity_grid)
                                
                                best_val = -1.0
                                best_j = None
                                # for j in remaining_unvisited:
                                for j in S:
                                    # S_next = tuple(sorted(list(S) + [i]))
                                    S_next = tuple(sorted(set(S) - {j}))
                                    # val = sum(prob * min(current_beta, dp_table.get((t + 1, next_c, v, j, remaining_unvisited), 0))
                                    val = sum(prob * min(current_beta, dp_table.get((t + 1, next_c, v, j, S_next), 0))
                                              for v, prob in joint_dist[j].items())
                                    if val > best_val:
                                        best_val = val
                                        best_j = j
                                
                                dp_table[(t, c, d, i, S)] = best_val
                                alloc_table[(t, c, d, i, S)] = alloc
                                route_table[(t, c, d, i, S)] = [best_j]
                                
                            elif algorithm == "manshadi_ppa":
                                for beta in beta_grid:
                                    beta = round_to_grid(beta, beta_grid)
                                    alloc = min(d * c / (d + future_mean), d)
                                    next_c = round_to_grid(c - alloc, capacity_grid)
                                    next_beta = round_to_grid(min(current_beta, beta), beta_grid)
                                    
                                    best_val = -1.0
                                    best_j = None
                                    for j in S:
                                        S_next = tuple(sorted(set(S) - {j}))
                                        val = sum(prob * dp_table.get((t + 1, next_beta, next_c, v, j, S_next), 0)
                                                  for v, prob in joint_dist[j].items())
                                        if val > best_val:
                                            best_val = val
                                            best_j = j
                                            
                                    dp_table[(t, beta, c, d, i, S)] = best_val
                                    alloc_table[(t, beta, c, d, i, S)] = alloc
                                    route_table[(t, beta, c, d, i, S)] = [best_j]
                                    
    return dp_table, alloc_table, route_table

D1 = {1: 1/5, 2: 1/5, 3: 1/5, 4: 1/5, 5: 1/5}
D2 = {1: 1/10, 2: 1/5, 3: 2/5, 4: 1/5, 5: 1/10}
D3 = {1: 2/5, 2: 3/10, 3: 1/5, 4: 7/100, 5: 3/100}
D4 = {1: 3/100, 2: 7/100, 3: 1/5, 4: 3/10, 5: 2/5}
D5 = {1: 2/5, 2: 3/40, 3: 1/20, 4: 3/40, 5: 2/5}
D6 = {1: 3/10, 2: 2/5, 3: 1/5, 4: 7/100, 5: 3/100}
D7 = {1: 1/5, 2: 7/100, 3: 3/100, 4: 3/10, 5: 2/5}
D8 = {1: 2/5, 2: 3/100, 3: 3/10, 4: 7/100, 5: 1/5}
distr_list = [D1, D2, D3, D4, D5, D6, D7, D8]
horizon = 3
# horizon = 3
algorithms = ["haiqing_ppa", "manshadi_ppa"]
settings = ["random"]
# settings = ["same_var", "same_mean", "same_CV", "random"]

for setting in settings:
    if setting == "same_var":
        shifts = [0.5, 1.0]
        type_range = range(len(distr_list))
    elif setting == "same_mean":
        mean = 3.5
        shifts = [mean]
        new_list = same_mean_generator(distr_list, mean)
        combinations = list(itertools.combinations(new_list, horizon))
        type_range = range(len(combinations))
    elif setting == "same_CV":
        CV_val = 0.35140452079103174 
        shifts = [CV_val]
        new_list = same_CV_generator(distr_list, CV_val)
        combinations = list(itertools.combinations(new_list, horizon))
        type_range = range(len(combinations))
    elif setting == "random":
        shifts = [0]
        # Use all combinations of horizon distributions from distr_list
        combinations = list(itertools.combinations(distr_list, horizon))
        type_range = range(2, len(combinations))

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
            for node in F:
                d_max += max(list(node.keys()))
                d_mean += sum(value * prob for value, prob in node.items())
            
            dynamic_results = {
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
                "routing": [],
                "sample_paths": [],
            }
            auto_step_size = 0.1
            beta_step_size = 0.01
            beta_grid = np.round(np.arange(0, 1 + 1e-10, beta_step_size), count_decimals(beta_step_size))
            capacity_list = np.arange(0.1, (d_max / d_mean) + 0.1, 0.1)
            c_max = np.max(capacity_list) * d_mean
            capacity_grid = np.round(np.arange(0, c_max + 1e-10, auto_step_size), count_decimals(auto_step_size))
            for algorithm in algorithms:
                for capacity_level in capacity_list:
                    c1 = round_to_grid(capacity_level * d_mean, capacity_grid)
                    
                    if algorithm == "haiqing_ppa":
                        subdir = "dynamic_ppa_same_var" if setting == "same_var" else ("dynamic_ppa_same_mean" if setting == "same_mean" else ("dynamic_ppa_same_CV" if setting == "same_CV" else "dynamic_ppa_random"))
                        path = os.path.join("./Fair_Sequential_New_DP/src/haiqing_dp_table", subdir)
                        os.makedirs(path, exist_ok=True)
                        if setting == "same_var":
                            file_path = os.path.join(path, f'type_{i}_shift_{shift}_horizon_{horizon}.csv')
                        elif setting == "same_mean":
                            file_path = os.path.join(path, f'type_{i}_mean_{shift}_horizon_{horizon}.csv')
                        elif setting == "random":
                            file_path = os.path.join(path, f'type_{i}_random_horizon_{horizon}.csv')
                        else: # same_CV
                            file_path = os.path.join(path, f'type_{i}_CV_{shift}_horizon_{horizon}.csv')
                        
                        try:
                            dp_table = {}
                            alloc_table = {}
                            route_table = {}
                            with open(file_path, mode='r', newline='') as file:
                                reader = csv.DictReader(file)
                                for row in reader:
                                    state = ast.literal_eval(row["state"])
                                    dp_table[state] = float(row["value"])
                                    alloc_table[state] = float(row["allocation"])
                                    route_table[state] = ast.literal_eval(row["routing"])
                        except FileNotFoundError:
                            dp_results = {"state": [], "value": [], "allocation": [], "routing": []}
                            dp_table, alloc_table, route_table = solve_dynamic_ppa(F, horizon, capacity_grid, beta_grid, algorithm)
                            for state in dp_table.keys():
                                dp_results["state"].append(state)
                                dp_results["value"].append(dp_table[state])
                                dp_results["allocation"].append(alloc_table[state])
                                dp_results["routing"].append(route_table[state])
                            df = pd.DataFrame(dp_results)
                            df.to_csv(file_path, index=False)
                    elif algorithm == "manshadi_ppa":
                        subdir = "dynamic_ppa_same_var" if setting == "same_var" else (
                            "dynamic_ppa_same_mean" if setting == "same_mean" else (
                                "dynamic_ppa_same_CV" if setting == "same_CV" else "dynamic_ppa_random"))
                        path = os.path.join("./Fair_Sequential_New_DP/src/manshadi_dp_table", subdir)
                        os.makedirs(path, exist_ok=True)
                        if setting == "same_var":
                            file_path = os.path.join(path, f'type_{i}_shift_{shift}_horizon_{horizon}.csv')
                        elif setting == "same_mean":
                            file_path = os.path.join(path, f'type_{i}_mean_{shift}_horizon_{horizon}.csv')
                        elif setting == "random":
                            file_path = os.path.join(path, f'type_{i}_random_horizon_{horizon}.csv')
                        else:  # same_CV
                            file_path = os.path.join(path, f'type_{i}_CV_{shift}_horizon_{horizon}.csv')
                        
                        try:
                            dp_table = {}
                            alloc_table = {}
                            route_table = {}
                            with open(file_path, mode='r', newline='') as file:
                                reader = csv.DictReader(file)
                                for row in reader:
                                    state = ast.literal_eval(row["state"])
                                    dp_table[state] = float(row["value"])
                                    alloc_table[state] = float(row["allocation"])
                                    route_table[state] = ast.literal_eval(row["routing"])
                        except FileNotFoundError:
                            dp_results = {"state": [], "value": [], "allocation": [], "routing": []}
                            dp_table, alloc_table, route_table = solve_dynamic_ppa(F, horizon, capacity_grid, beta_grid, algorithm)
                            for state in dp_table.keys():
                                dp_results["state"].append(state)
                                dp_results["value"].append(dp_table[state])
                                dp_results["allocation"].append(alloc_table[state])
                                dp_results["routing"].append(route_table[state])
                            df = pd.DataFrame(dp_results)
                            df.to_csv(file_path, index=False)
                    else:
                        print("Wrong algorithm input. Please check.")
                        break
                    
                    list_path, list_alloc, list_prob, list_route = dfs((0, c1), F, algorithm, dp_table, alloc_table, route_table, horizon, capacity_grid, beta_grid)
                    
                    first_node = choose_first_node(c1, F, algorithm, dp_table, horizon)
                    haiqing_obj = calculate_dynamic_haiqing_time_zero_objective(c1, first_node, F[first_node], dp_table,
                                                                                algorithm, horizon)
                    ex_post_min_fr = calculate_ex_post(det_min_fill_rate, list_alloc, list_path, list_prob)
                    ex_post_preference = calculate_ex_post(det_preference, list_alloc, list_path, list_prob)
                    ex_post_envy = calculate_ex_post(det_envy, list_alloc, list_path, list_prob)
                    expected_min_fr, ex_ante_preference = ex_ante_fr(list_alloc, list_path, list_prob)
                    ex_ante_envy = calculate_ex_ante_envy(list_alloc, list_path, list_prob)
                    expected_waste = calculate_expected_waste(c1, list_alloc, list_prob)
                    dynamic_results["algorithm"].append(algorithm)
                    dynamic_results["discretized_distribution"].append(F)
                    dynamic_results["horizon"].append(horizon)
                    dynamic_results["supply_divided_by_mean_demand"].append(capacity_level)
                    dynamic_results["initial_supply"].append(c1)
                    dynamic_results["haiqing_obj"].append(haiqing_obj)
                    dynamic_results["one_time_fairness"].append(ex_post_min_fr)
                    dynamic_results["long_time_fairness"].append(expected_min_fr)
                    dynamic_results["ex_ante_preference"].append(ex_ante_preference)
                    dynamic_results["ex_post_preference"].append(ex_post_preference)
                    dynamic_results["ex_ante_envy"].append(ex_ante_envy)
                    dynamic_results["ex_post_envy"].append(ex_post_envy)
                    dynamic_results["expected_waste"].append(expected_waste)
                    dynamic_results["sample_paths"].append(list_path)
                    dynamic_results["allocations"].append(list_alloc)
                    dynamic_results["routing"].append(list_route)

            df = pd.DataFrame(dynamic_results)
            res_dir = f"./dynamic_ppa_{setting}_results"
            os.makedirs(res_dir, exist_ok=True)
            if setting == "same_var":
                res_file = f"dynamic_ppa_type{i}_shift_{shift}_horizon_{horizon}.csv"
            elif setting == "same_mean":
                res_file = f"dynamic_ppa_type{i}_mean_{shift}_horizon_{horizon}.csv"
            elif setting == "random":
                res_file = f"dynamic_ppa_type_{i}_random_horizon_{horizon}.csv"
            else:
                res_file = f"dynamic_ppa_type{i}_CV_{shift}_horizon_{horizon}.csv"
            df.to_csv(os.path.join(res_dir, res_file), index=False)
