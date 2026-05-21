import numpy as np
import pandas as pd
import scipy

from math import gcd
from functools import reduce
import itertools

def cal_gcd(arr):
    return reduce(gcd, arr)

def round_to_grid(number, grid):
    return min(grid, key=lambda x: abs(x - number))

def calculate_haiqing_obj(x, dp_table, state, future_dist, c_grid):
    t, c, d = state
    obj = 0
    for future_d in list(future_dist.keys()):
        # print(f'obj is {obj} before adding expectation with future_d={future_d}')
        obj += future_dist[future_d] * min(x / d, dp_table[(t + 1, round_to_grid(c - x, c_grid), future_d)])
        # print(f'obj is {obj} after adding expectation with current fr {x/d} and future fr={dp_table[(t + 1, round(c - x, 1), future_d)]}')
    return obj

def calculate_dynamic_haiqing_obj(x, j, dp_table, state, future_dist, c_grid):
    t, c, d, i, S = state
    obj = 0
    for future_d in list(future_dist.keys()):
        # print(f'obj is {obj} before adding expectation with future_d={future_d}')
        obj += future_dist[future_d] * min(x / d, dp_table[(t + 1, round_to_grid(c - x, c_grid), future_d, j, tuple(i for i in S if i != j))])
        # print(f'obj is {obj} after adding expectation with current fr {x/d} and future fr={dp_table[(t + 1, round(c - x, 1), future_d)]}')
    return obj

def equating_haiqing(dp_table, state, step_size, future_demands, c_grid):
    t, c, d = state
    constraint = min(c, d)
    d_min = min(future_demands)
    # print(f"d min = {d_min}")
    d_max = max(future_demands)
    # print(f"d max = {d_max}")
    upper_bound = 0
    lower_bound = 0
    x_search_space = np.arange(0, constraint + 1e-10, step_size)
    # print(f'x_search_space = {x_search_space}')
    for x in x_search_space:
        if x >= dp_table[(t + 1, round_to_grid(c - x, c_grid), d_max)]*d:
            lower_bound = x
            # print(f'lower_bound = {lower_bound}')
            break
    for x in reversed(x_search_space):
        if x <= dp_table[(t + 1, round_to_grid(c - x, c_grid), d_min)]*d:
            upper_bound = min(x + step_size, constraint)
            # print(f'upper_bound = {upper_bound}')
            break
    # print("lower_bound", lower_bound, "upper_bound", upper_bound)
    if lower_bound >= upper_bound:
        return x_search_space
    else:
        return np.arange(lower_bound, upper_bound + 1e-10, step_size)

# def equating_haiqing(dp_table, state, step_size, future_demands, c_grid):
#     '''binary search with checking criterion equating property'''
#     # print(f'Begin equating search at state {state}...')
#     t, c, d = state
#     upper_bound = min(c, d)
#     # if upper_bound == c:
#     #     x_search_space = np.arange(0, upper_bound, step_size)
#     # else:
#     #     x_search_space = np.arange(0, upper_bound + 1e-10, step_size)
#     x_search_space = np.arange(0, upper_bound + 1e-10, step_size)
#     # print(f'x space is {x_search_space}')
#     l = len(x_search_space)
#     candidates = set()
#     for future_d in future_demands:
#         # find the root interval
#         left = 0
#         right = l-1
#         # print(f'right end is {x_search_space[right]}')
#         # print(f'state is {(t + 1, round_to_grid(c - x_search_space[right], c_grid), future_d)}')
#         # print(f'{dp_table[(t + 1, round_to_grid(c - x_search_space[right], c_grid), future_d)]}')
#         zero_not_found = True
#         if round(dp_table[(t + 1, round_to_grid(c - x_search_space[right], c_grid), future_d)], 5) == 1:
#             candidates.add(x_search_space[right])
#             zero_not_found = False
#         else:
#             while left + 1 < right:
#                 mid = int((left + right)/2)
#                 x = round_to_grid(x_search_space[mid], c_grid)
#                 # print(f'x is {x}, and c is {c} with c - x {c-x}')
#                 if round(dp_table[(t + 1, round_to_grid(c - x, c_grid), future_d)]*d - x, 5) == 0:
#                     candidates.add(x)
#                     zero_not_found = False
#                     break
#                 elif dp_table[(t + 1, round_to_grid(c - x, c_grid), future_d)]*d - x > 0:
#                     left = mid
#                 else:
#                     right = mid
#                 # print(f'left end is {x_search_space[left]} with right {x_search_space[right]}.')
#         if zero_not_found:
#             candidates.add(round_to_grid(x_search_space[left], c_grid))
#             candidates.add(round_to_grid(x_search_space[right], c_grid))
#     return sorted(list(candidates))

# Use dictionary to store obejctives and allocations
def solve_homogeneous_haiqing_DP(dist, horizon, capacity_grid, auto_step_size):
    demands = list(dist.keys())
    # value function table
    dp_table = {}
    # allocation table
    alloc_table = {}
    for t in range(horizon, 0, -1):
        # print(f'Begin filling the table for n = {t}...')
        for c in capacity_grid:
            c = round_to_grid(c, capacity_grid)
            # print(f'Capacity is {c}')
            for d in demands:
                # print(f'Demands is {d}')
                if t == horizon:
                    dp_table[(t, c, d)] = min(c / d, 1)
                    alloc_table[(t, c, d)] = min(c, d)
                else:
                    dp_table[(t, c, d)] = 0
                    # if c <= d:
                    #     alloc_candidates = np.arange(0, c, auto_step_size)
                    # else:
                    #     alloc_candidates = np.arange(0, d + 1e-10, auto_step_size)
                    alloc_candidates = equating_haiqing(dp_table, (t, c, d), auto_step_size, demands, capacity_grid)
                    # print(f'At state {(t, c, d)}, candidates are {alloc_candidates}.')
                    for x in alloc_candidates:
                        alloc_obj = calculate_haiqing_obj(x, dp_table, (t, c, d), dist, capacity_grid)
                        if round(alloc_obj, 7) >= round(dp_table[(t, c, d)], 7):
                            dp_table[(t, c, d)] = alloc_obj
                            alloc_table[(t, c, d)] = x
                # print(f'The objective at state {(t, c, d)} is {dp_table[(t, c, d)]}, with the optimal allocation {alloc_table[(t, c, d)]}.')
    return dp_table, alloc_table

def brute_force_solve_homogeneous_haiqing_DP(dist, horizon, capacity_grid, auto_step_size):
    demands = list(dist.keys())
    # value function table
    dp_table = {}
    # allocation table
    alloc_table = {}
    for t in range(horizon, 0, -1):
        # print(f'Begin filling the table for n = {t}...')
        for c in capacity_grid:
            c = round_to_grid(c, capacity_grid)
            # print(f'Capacity is {c}')
            for d in demands:
                # print(f'Demands is {d}')
                if t == horizon:
                    dp_table[(t, c, d)] = min(c / d, 1)
                    alloc_table[(t, c, d)] = min(c, d)
                else:
                    dp_table[(t, c, d)] = 0
                    upper_bound = min(c, d)
                    alloc_candidates = np.arange(0, upper_bound + 1e-10, auto_step_size)
                    # if c <= d:
                    #     alloc_candidates = np.arange(0, c, auto_step_size)
                    # else:
                    #     alloc_candidates = np.arange(0, d + 1e-10, auto_step_size)
                    # alloc_candidates = equating_haiqing(dp_table, (t, c, d), auto_step_size, demands, capacity_grid)
                    # print(f'At state {(t, c, d)}, candidates are {alloc_candidates}.')
                    for x in alloc_candidates:
                        alloc_obj = calculate_haiqing_obj(x, dp_table, (t, c, d), dist, capacity_grid)
                        if alloc_obj >= dp_table[(t, c, d)]:
                            dp_table[(t, c, d)] = alloc_obj
                            alloc_table[(t, c, d)] = round_to_grid(x, capacity_grid)
                # print(f'The objective at state {(t, c, d)} is {dp_table[(t, c, d)]}, with the optimal allocation {alloc_table[(t, c, d)]}.')
    return dp_table, alloc_table

def equate_solve_homogeneous_haiqing_DP(dist, horizon, capacity_grid, auto_step_size):
    demands = list(dist.keys())
    # value function table
    dp_table = {}
    # allocation table
    alloc_table = {}
    for t in range(horizon, 0, -1):
        # print(f'Begin filling the table for n = {t}...')
        for c in capacity_grid:
            c = round_to_grid(c, capacity_grid)
            # print(f'Capacity is {c}')
            for d in demands:
                # print(f'Demands is {d}')
                if t == horizon:
                    dp_table[(t, c, d)] = min(c / d, 1)
                    alloc_table[(t, c, d)] = min(c, d)
                else:
                    dp_table[(t, c, d)] = 0
                    # if c <= d:
                    #     alloc_candidates = np.arange(0, c, auto_step_size)
                    # else:
                    #     alloc_candidates = np.arange(0, d + 1e-10, auto_step_size)
                    alloc_candidates = equating_haiqing(dp_table, (t, c, d), auto_step_size, demands, capacity_grid)
                    # print(f'At state {(t, c, d)}, candidates are {alloc_candidates}.')
                    for x in alloc_candidates:
                        alloc_obj = calculate_haiqing_obj(x, dp_table, (t, c, d), dist, capacity_grid)
                        if round(alloc_obj, 9) >= round(dp_table[(t, c, d)],9):
                            dp_table[(t, c, d)] = alloc_obj
                            alloc_table[(t, c, d)] = round_to_grid(x, capacity_grid)
                # print(f'The objective at state {(t, c, d)} is {dp_table[(t, c, d)]}, with the optimal allocation {alloc_table[(t, c, d)]}.')
    return dp_table, alloc_table

def base_solve_heterogeneous_haiqing_DP(joint_dist, horizon, capacity_grid, auto_step_size):
    # value function table
    dp_table = {}
    # allocation table
    alloc_table = {}
    for t in range(horizon, 0, -1):
        # print(f'Begin filling the table for n = {t}...')
        current_demands = list(joint_dist[t-1].keys())
        # print(f'Current distribution: {joint_dist[t-1]}')
        for c in capacity_grid:
            c = round_to_grid(c, capacity_grid)
            # print(f'Capacity is {c}')
            for d in current_demands:
                # print(f'Demands is {d}')
                if t == horizon:
                    dp_table[(t, c, d)] = min(c / d, 1)
                    alloc_table[(t, c, d)] = min(c, d)
                else:
                    dp_table[(t, c, d)] = 0
                    # future_demands = list(joint_dist[t].keys())
                    # if c <= d:
                    #     alloc_candidates = np.arange(0, c, auto_step_size)
                    # else:
                    #     alloc_candidates = np.arange(0, d + 1e-10, auto_step_size)
                    # alloc_candidates = equating_haiqing(dp_table, (t, c, d), auto_step_size, future_demands, capacity_grid)
                    upper_bound = min(c, d)
                    alloc_candidates = np.arange(0, upper_bound + 1e-10, auto_step_size)
                    # print(f'At state {(t, c, d)}, candidates are {alloc_candidates}.')
                    for x in alloc_candidates:
                        alloc_obj = calculate_haiqing_obj(x, dp_table, (t, c, d), joint_dist[t], capacity_grid)
                        if alloc_obj >= dp_table[(t, c, d)]:
                            dp_table[(t, c, d)] = alloc_obj
                            alloc_table[(t, c, d)] = x
                # print(f'The objective at state {(t, c, d)} is {dp_table[(t, c, d)]}, with the optimal allocation {alloc_table[(t, c, d)]}.')
    return dp_table, alloc_table

def solve_dynamic_haiqing_DP(joint_dist, horizon, capacity_grid, auto_step_size):
    # value function table
    dp_table = {}
    # allocation table
    alloc_table = {}
    # routing table
    route_table = {}
    for t in range(horizon, 0, -1):
        print(f'Begin filling the table for n = {t}...')
        # Generate candidates for the remaining unvisited nodes
        for S in list(itertools.combinations(range(horizon), horizon - t)):
            print(f'Remaining unvisited set is {S}')
            pool = list(set(range(horizon)) - set(S))
            for i in pool:
                print(f'We are at node {i}')
                current_demands = joint_dist[i].keys()
                for c in capacity_grid:
                    c = round_to_grid(c, capacity_grid)
                    print(f'Capacity is {c}')
                    for d in current_demands:
                        print(f'Demands is {d}')
                        if t == horizon:
                            dp_table[(t, c, d, i, S)] = min(c / d, 1)
                            alloc_table[(t, c, d, i, S)] = min(c, d)
                            route_table[(t, c, d, i, S)] = ()
                        else:
                            dp_table[(t, c, d, i, S)] = 0
                            route_table[(t, c, d, i, S)] = []
                            upper_bound = min(c, d)
                            alloc_candidates = np.arange(0, upper_bound + 1e-10, auto_step_size)
                            for j in S:
                                # current_route_obj = 0
                                for x in alloc_candidates:
                                    alloc_obj = calculate_dynamic_haiqing_obj(x, j, dp_table, (t, c, d, i, S), joint_dist[j], capacity_grid)
                                    if alloc_obj >= dp_table[(t, c, d, i, S)]:
                                        dp_table[(t, c, d, i, S)] = alloc_obj
                                        alloc_table[(t, c, d, i, S)] = x
                                        route_table[(t, c, d, i, S)] = [j]
                                # if current_route_obj > dp_table[(t, c, d, i, S)]:
                                #     dp_table[(t, c, d, i, S)] = current_route_obj
                                #     alloc_table[(t, c, d, i, S)] = current_alloc
                                #     route_table[(t, c, d, i, S)] = [j]
                                # elif current_route_obj == dp_table[(t, c, d, i, S)]:
                                #     if j not in route_table[(t, c, d, i, S)]:
                                #         route_table[(t, c, d, i, S)].append(j)
                            print(f'The objective at state {(t, c, d, i, S)} is {dp_table[(t, c, d, i, S)]}, with the optimal allocation {alloc_table[(t, c, d, i, S)]} and next node {route_table[(t, c, d, i, S)]}.')
    return dp_table, alloc_table, route_table


def count_decimals(number):
    s = str(number)
    if '.' in s:
        return len(s.split('.')[1].rstrip('0'))
    return 0
import csv
import ast

if __name__ == '__main__':
    D1 = {1: 1 / 5, 2: 1 / 5, 3: 1 / 5, 4: 1 / 5, 5: 1 / 5}
    D2 = {1: 1 / 10, 2: 1 / 5, 3: 2 / 5, 4: 1 / 5, 5: 1 / 10}
    D3 = {1: 2 / 5, 2: 3 / 10, 3: 1 / 5, 4: 7 / 100, 5: 3 / 100}
    D4 = {1: 3 / 100, 2: 7 / 100, 3: 1 / 5, 4: 3 / 10, 5: 2 / 5}
    D5 = {1: 2 / 5, 2: 3 / 40, 3: 1 / 20, 4: 3 / 40, 5: 2 / 5}
    D6 = {1: 3 / 10, 2: 2 / 5, 3: 1 / 5, 4: 7 / 100, 5: 3 / 100}
    D7 = {1: 1 / 5, 2: 7 / 100, 3: 3 / 100, 4: 3 / 10, 5: 2 / 5}
    D8 = {1: 2 / 5, 2: 3 / 100, 3: 3 / 10, 4: 7 / 100, 5: 1 / 5}
    # F = [D1, D2, D3, D4, D5, D6, D7, D8]
    horizon = 3
    F = [D1, D2, D3]
    auto_step_size = 0.1
    # beta_step_size = 0.01
    # beta_grid = np.round(np.arange(0, 1 + 1e-10, beta_step_size), count_decimals(beta_step_size))
    capacity_grid = np.round(np.arange(0, 15 + 1e-10, auto_step_size), count_decimals(auto_step_size))
    solve_dynamic_haiqing_DP(F, horizon, capacity_grid, auto_step_size)
    # algorithm = ["equating", "brute-force"]
    # for i in range(len(F)):
    #     D = F[i]
    #     demands = list(D.keys())
    #     d_max = max(demands)
    #     d_mean = np.mean(demands)
    #     horizon = 3
    #     auto_step_size = 0.1
    #     # beta_step_size = 0.01
    #     # beta_grid = np.round(np.arange(0, 1 + 1e-10, beta_step_size), count_decimals(beta_step_size))
    #     # step_size = 0.1
    #     # auto_step_size = min(step_size, cal_gcd(demands))
    #     capacity_list = np.arange(0.1, (d_max / d_mean) + 0.1, 0.1)
    #     c_max = np.max(capacity_list) * d_mean * horizon
    #     capacity_grid = np.round(np.arange(0, c_max + 1e-10, auto_step_size), count_decimals(auto_step_size))
    #     # brute-force search
    #     try:
    #         dp_table_brute_force = {}
    #         alloc_table_brute_force = {}
    #         with open(f'./haiqing_dp_table/brute_force_discrete_type_{i}.csv', mode='r', newline='') as file:
    #             # with open(f'./manshadi_dp_table/discrete_type_{i}_shift_{distance}.csv', mode='r', newline='') as file:
    #             reader = csv.DictReader(file)
    #             for row in reader:
    #                 state = ast.literal_eval(row["state"])
    #                 dp_table_brute_force[state] = float(row["value"])
    #                 alloc_table_brute_force[state] = float(row["allocation"])
    #     except FileNotFoundError:
    #         dp_results = {"state": [], "value": [], "allocation": []}
    #         dp_table_brute_force, alloc_table_brute_force = brute_force_solve_homogeneous_haiqing_DP(D, horizon,
    #                                                                                                   capacity_grid,
    #                                                                                                   auto_step_size)
    #         for state in dp_table_brute_force.keys():
    #             dp_results["state"].append(state)
    #             dp_results["value"].append(dp_table_brute_force[state])
    #             dp_results["allocation"].append(alloc_table_brute_force[state])
    #         df = pd.DataFrame(dp_results)
    #         df.to_csv(f'./haiqing_dp_table/brute_force_discrete_type_{i}.csv', index=False)
    #     # reduced search space
    #     try:
    #         dp_table_equate = {}
    #         alloc_table_equate = {}
    #         with open(f'./haiqing_dp_table/revised_euqate_discrete_type_{i}.csv', mode='r', newline='') as file:
    #             # with open(f'./manshadi_dp_table/discrete_type_{i}_shift_{distance}.csv', mode='r', newline='') as file:
    #             reader = csv.DictReader(file)
    #             for row in reader:
    #                 state = ast.literal_eval(row["state"])
    #                 dp_table_equate[state] = float(row["value"])
    #                 alloc_table_equate[state] = float(row["allocation"])
    #     except FileNotFoundError:
    #         dp_results = {"state": [], "value": [], "allocation": []}
    #         dp_table_equate, alloc_table_equate = equate_solve_homogeneous_haiqing_DP(D, horizon, capacity_grid,
    #                                                                                         auto_step_size)
    #         for state in dp_table_equate.keys():
    #             dp_results["state"].append(state)
    #             dp_results["value"].append(dp_table_equate[state])
    #             dp_results["allocation"].append(alloc_table_equate[state])
    #         df = pd.DataFrame(dp_results)
    #         df.to_csv(f'./haiqing_dp_table/revised_equate_discrete_type_{i}.csv', index=False)
    #     print(f'distribution type {i}: {alloc_table_brute_force == alloc_table_equate}')
#     D1 = {1:0.5, 2:0.5}
#     D2 = {6:0.5, 7:0.5}
#     F = [D1, D2]
#     horizon = 2
#     capacity_grid = np.round(np.arange(0.01, 10 + 1e-10, 0.01), count_decimals(0.01))
#     dp_table, alloc_table = solve_heterogeneous_haiqing_DP(F, horizon, capacity_grid, 0.01)
#     print(dp_table)
    # import csv
    # import ast
    # dp_table = {}
    # alloc_table = {}
    # with open(f'./haiqing_dp_table/three_points_increasing_discrete_type_1.csv', mode='r', newline='') as file:
    #     reader = csv.DictReader(file)
    #     for row in reader:
    #         state = ast.literal_eval(row["state"])
    #         dp_table[state] = float(row["value"])
    #         alloc_table[state] = float(row["allocation"])
    # demands = [1, 2, 3]
    # d_max = max(demands)
    # d_mean = np.mean(demands)
    # horizon = 3
    # auto_step_size = 0.1
    # beta_step_size = 0.01
    # beta_grid = np.round(np.arange(beta_step_size, 1 + 1e-10, beta_step_size), count_decimals(beta_step_size))
    # # step_size = 0.1
    # # auto_step_size = min(step_size, cal_gcd(demands))
    # capacity_list = np.arange(0.1, (d_max/d_mean) + 0.1, 0.1)
    # c_max = np.max(capacity_list)*d_mean*horizon
    # capacity_grid = np.round(np.arange(auto_step_size, c_max + 1e-10, auto_step_size), count_decimals(auto_step_size))
    # alloc_candidates = equating_haiqing(dp_table, (1, 8.4, 1), 0.1, demands, capacity_grid)
    # print(alloc_candidates)

