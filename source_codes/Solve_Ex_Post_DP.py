import numpy as np
import pandas as pd
import scipy
import csv
import ast

from math import gcd
from functools import reduce
import itertools

def cal_gcd(arr):
    return reduce(gcd, arr)

def round_to_grid(number, grid):
    return min(grid, key=lambda x: abs(x - number))

def calculate_manshadi_obj(x, dp_table, state, dist, c_grid, beta_grid):
    t, beta, c, d = state
    obj = 0
    for future_d in list(dist.keys()):
        # print(f'obj is {obj} before adding expectation with future_d={future_d}')
        obj += dist[future_d]*dp_table[(t + 1, round_to_grid(min(beta, x / d), beta_grid), round_to_grid(c - x, c_grid), future_d)]
        # print(f'obj is {obj} after adding expectation with current fr {x/d} and future fr={dp_table[(t + 1, round(c - x, 1), future_d)]}')
    return obj

def calculate_dynamic_manshadi_obj(x, j, dp_table, state, dist, c_grid, beta_grid):
    t, beta, c, d, i, S = state
    obj = 0
    for future_d in list(dist.keys()):
        # print(f'obj is {obj} before adding expectation with future_d={future_d}')
        obj += dist[future_d]*dp_table[(t + 1, round_to_grid(min(beta, x / d), beta_grid), round_to_grid(c - x, c_grid), future_d, j, tuple(i for i in S if i != j))]
        # print(f'obj is {obj} after adding expectation with current fr {x/d} and future fr={dp_table[(t + 1, round(c - x, 1), future_d)]}')
    return obj

def calculate_manshadi_time_zero_objective(c, D, dp_table):
    Z0 = 0
    for d in D.keys():
        Z0 += D[d] * dp_table[(1, 1, c, d)]
    return Z0


def equating_manshadi(dp_table, state, step_size, demands, c_grid, beta_grid):
    '''binary search with checking criterion equating property'''
    # print(f'Begin equating search at state {state}...')
    t, beta, c, d = state
    constraint = min(c, d)
    d_min = min(demands)
    d_max = max(demands)
    upper_bound = 0
    lower_bound = 0
    x_search_space = np.arange(0, constraint + 1e-10, step_size)
    for x in x_search_space:
        if x >= d*dp_table[(t + 1, round_to_grid(min(x/d_max, beta), beta_grid), round_to_grid(c - x, c_grid), d_max)]:
            lower_bound = x
            break
    for x in reversed(x_search_space):
        if x <= d*dp_table[(t + 1, round_to_grid(min(x/d_min, beta), beta_grid), round_to_grid(c - x, c_grid), d_min)]:
            upper_bound = x
            break
    # print(f'x space is {x_search_space}')
    # l = len(x_search_space)
    # # candidates = set()
    # for future_d in [d_min, d_max]:
    #     # find the root interval
    #     left = 0
    #     right = l - 1
    #     zero_not_found = True
    #     if dp_table[(t + 1, round_to_grid(c - x_search_space[right], c_grid), future_d)] == 1:
    #         candidates.add(x_search_space[right])
    #         zero_not_found = False
    #     else:
    #         while left + 1 < right:
    #             mid = int((left + right)/2)
    #             x = round_to_grid(x_search_space[mid], c_grid)
    #             # print(f'x is {x}, and c is {c} with c - x {c-x}')
    #             if dp_table[(t + 1, round_to_grid(c - x, c_grid), future_d)]*d - x == 0:
    #                 candidates.add(x)
    #                 zero_not_found = False
    #                 break
    #             elif dp_table[(t + 1, round_to_grid(c - x, c_grid), future_d)]*d - x > 0:
    #                 left = mid
    #             else:
    #                 right = mid
    #             # print(f'left end is {left} with right {right}.')
    #     if zero_not_found:
    #         candidates.add(round_to_grid(x_search_space[left], c_grid))
    #         candidates.add(round_to_grid(x_search_space[right], c_grid))
    if lower_bound >= upper_bound:
        return x_search_space
    else:
        return np.arange(lower_bound, upper_bound + 1e-10, step_size)

# Use dictionary to store obejctives and allocations
'''
Note that this function solves the homogeneous problem with beta constraint. 
Not what we want since we want to avoid the waste.
'''
def solve_homogeneous_manshadi_DP(dist, horizon, beta_grid, capacity_grid, auto_step_size):
    demands = list(dist.keys())
    # value function table
    dp_table = {}
    # allocation table
    alloc_table = {}
    for t in range(horizon, 0, -1):
        print(f'Begin filling the table for n = {t}...')
        for beta in beta_grid:
            beta = round_to_grid(beta, beta_grid)
            for c in capacity_grid:
                c = round_to_grid(c, capacity_grid)
                print(f'Capacity is {c}')
                for d in demands:
                    print(f'Demands is {d}')
                    if t == horizon:
                        dp_table[(t, beta, c, d)] = min(c / d, beta)
                        alloc_table[(t, beta, c, d)] = min(c, d*beta)
                    else:
                        dp_table[(t, beta, c, d)] = 0
                        # if c > d*beta + 1e-5: # check this, can use EPS = 1e-10
                        #     alloc_candidates = np.arange(0, d*beta + 1e-10, auto_step_size)
                        # else:
                        #     alloc_candidates = np.arange(0, c - 1e-10, auto_step_size)
                        upper_bound = min(c, d)
                        alloc_candidates = np.arange(0, upper_bound + 1e-10, auto_step_size)
                        # alloc_candidates = equating_haiqing(dp_table, (t, c, d), auto_step_size, demands)
                        # print(f'At state {(t, beta, c, d)}, candidates are {alloc_candidates}.')
                        for x in alloc_candidates:
                            alloc_obj = calculate_manshadi_obj(x, dp_table, (t, beta, c, d), dist, capacity_grid, beta_grid)
                            if alloc_obj >= dp_table[(t, beta, c, d)]:
                                dp_table[(t, beta, c, d)] = alloc_obj
                                alloc_table[(t, beta, c, d)] = x
                    print(f'The objective at state {(t, beta, c, d)} is {dp_table[(t, beta, c, d)]}, with the optimal allocation {alloc_table[(t, beta, c, d)]}.')
    return dp_table, alloc_table

def revised_end_solve_homogeneous_manshadi_DP(dist, horizon, beta_grid, capacity_grid, auto_step_size):
    demands = list(dist.keys())
    # value function table
    dp_table = {}
    # allocation table
    alloc_table = {}
    for t in range(horizon, 0, -1):
        # print(f'Begin filling the table for n = {t}...')
        for beta in beta_grid:
            beta = round_to_grid(beta, beta_grid)
            for c in capacity_grid:
                c = round_to_grid(c, capacity_grid)
                # print(f'Capacity is {c}')
                for d in demands:
                    # print(f'Demands is {d}')
                    if t == horizon:
                        dp_table[(t, beta, c, d)] = min(c / d, beta)
                        alloc_table[(t, beta, c, d)] = min(c, d)
                    else:
                        dp_table[(t, beta, c, d)] = 0
                        # if c > d*beta + 1e-5: # check this, can use EPS = 1e-10
                        #     alloc_candidates = np.arange(0, d*beta + 1e-10, auto_step_size)
                        # else:
                        #     alloc_candidates = np.arange(0, c - 1e-10, auto_step_size)
                        upper_bound = min(c, d)
                        alloc_candidates = np.arange(0, upper_bound + 1e-10, auto_step_size)
                        # alloc_candidates = equating_haiqing(dp_table, (t, c, d), auto_step_size, demands)
                        # print(f'At state {(t, beta, c, d)}, candidates are {alloc_candidates}.')
                        for x in alloc_candidates:
                            alloc_obj = calculate_manshadi_obj(x, dp_table, (t, beta, c, d), dist, capacity_grid, beta_grid)
                            if alloc_obj >= dp_table[(t, beta, c, d)]:
                                dp_table[(t, beta, c, d)] = alloc_obj
                                alloc_table[(t, beta, c, d)] = x
                    # print(f'The objective at state {(t, beta, c, d)} is {dp_table[(t, beta, c, d)]}, with the optimal allocation {alloc_table[(t, beta, c, d)]}.')
    return dp_table, alloc_table

def base_heterogeneous_manshadi_DP(joint_dist, horizon, beta_grid, capacity_grid, auto_step_size):
    # value function table
    dp_table = {}
    # allocation table
    alloc_table = {}
    for t in range(horizon, 0, -1):
        print(f'Begin filling the table for n = {t}...')
        current_demands = joint_dist[t-1].keys()
        for beta in beta_grid:
            beta = round_to_grid(beta, beta_grid)
            for c in capacity_grid:
                c = round_to_grid(c, capacity_grid)
                print(f'Capacity is {c}')
                for d in current_demands:
                    print(f'Demands is {d}')
                    if t == horizon:
                        dp_table[(t, beta, c, d)] = min(c / d, beta)
                        alloc_table[(t, beta, c, d)] = min(c, d)
                    else:
                        # future_demands = joint_dist[t].keys()
                        dp_table[(t, beta, c, d)] = 0
                        # if c > d*beta + 1e-5: # check this, can use EPS = 1e-10
                        #     alloc_candidates = np.arange(0, d*beta + 1e-10, auto_step_size)
                        # else:
                        #     alloc_candidates = np.arange(0, c - 1e-10, auto_step_size)
                        upper_bound = min(c, d)
                        alloc_candidates = np.arange(0, upper_bound + 1e-10, auto_step_size)
                        # alloc_candidates = equating_haiqing(dp_table, (t, c, d), auto_step_size, demands)
                        # print(f'At state {(t, beta, c, d)}, candidates are {alloc_candidates}.')
                        for x in alloc_candidates:
                            alloc_obj = calculate_manshadi_obj(x, dp_table, (t, beta, c, d), joint_dist[t], capacity_grid, beta_grid)
                            if alloc_obj >= dp_table[(t, beta, c, d)]:
                                dp_table[(t, beta, c, d)] = alloc_obj
                                alloc_table[(t, beta, c, d)] = x
                    print(f'The objective at state {(t, beta, c, d)} is {dp_table[(t, beta, c, d)]}, with the optimal allocation {alloc_table[(t, beta, c, d)]}.')
    return dp_table, alloc_table

def revised_equating_solve_homogeneous_manshadi_DP(dist, horizon, beta_grid, capacity_grid, auto_step_size):
    demands = list(dist.keys())
    # value function table
    dp_table = {}
    # allocation table
    alloc_table = {}
    for t in range(horizon, 0, -1):
        # print(f'Begin filling the table for n = {t}...')
        for beta in beta_grid:
            beta = round_to_grid(beta, beta_grid)
            for c in capacity_grid:
                c = round_to_grid(c, capacity_grid)
                # print(f'Capacity is {c}')
                for d in demands:
                    # print(f'Demands is {d}')
                    if t == horizon:
                        dp_table[(t, beta, c, d)] = min(c / d, beta)
                        alloc_table[(t, beta, c, d)] = min(c, d)
                    else:
                        dp_table[(t, beta, c, d)] = 0
                        # if c > d*beta + 1e-5: # check this, can use EPS = 1e-10
                        #     alloc_candidates = np.arange(0, d*beta + 1e-10, auto_step_size)
                        # else:
                        #     alloc_candidates = np.arange(0, c - 1e-10, auto_step_size)
                        upper_bound = min(c, d)
                        # alloc_candidates = np.arange(0, upper_bound + 1e-10, auto_step_size)
                        alloc_candidates = equating_manshadi(dp_table, (t, beta, c, d), auto_step_size, demands, capacity_grid, beta_grid)
                        # print(f'At state {(t, beta, c, d)}, candidates are {alloc_candidates}.')
                        for x in alloc_candidates:
                            alloc_obj = calculate_manshadi_obj(x, dp_table, (t, beta, c, d), dist, capacity_grid, beta_grid)
                            if alloc_obj >= dp_table[(t, beta, c, d)]:
                                dp_table[(t, beta, c, d)] = alloc_obj
                                alloc_table[(t, beta, c, d)] = x
                    # print(f'The objective at state {(t, beta, c, d)} is {dp_table[(t, beta, c, d)]}, with the optimal allocation {alloc_table[(t, beta, c, d)]}.')
    return dp_table, alloc_table

def solve_dynamic_manshadi_DP(joint_dist, horizon, beta_grid, capacity_grid, auto_step_size):
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
                for beta in beta_grid:
                    beta = round_to_grid(beta, beta_grid)
                    for c in capacity_grid:
                        c = round_to_grid(c, capacity_grid)
                        print(f'Capacity is {c}')
                        for d in current_demands:
                            print(f'Demands is {d}')
                            if t == horizon:
                                dp_table[(t, beta, c, d, i, S)] = min(c / d, beta)
                                alloc_table[(t, beta, c, d, i, S)] = min(c, d)
                                route_table[(t, beta, c, d, i, S)] = []
                            else:
                                dp_table[(t, beta, c, d, i, S)] = 0
                                route_table[(t, beta, c, d, i, S)] = []
                                upper_bound = min(c, d)
                                alloc_candidates = np.arange(0, upper_bound + 1e-10, auto_step_size)
                                for j in S:
                                    # current_route_obj = 0
                                    for x in alloc_candidates:
                                        alloc_obj = calculate_dynamic_manshadi_obj(x, j, dp_table, (t, beta, c, d, i, S), joint_dist[j], capacity_grid, beta_grid)
                                        if alloc_obj >= dp_table[(t, beta, c, d, i, S)]:
                                            dp_table[(t, beta, c, d, i, S)] = alloc_obj
                                            alloc_table[(t, beta, c, d, i, S)] = x
                                            route_table[(t, beta, c, d, i, S)] = [j]
                                    # if current_route_obj > dp_table[(t, beta, c, d, i, S)]:
                                    #     dp_table[(t, beta, c, d, i, S)] = current_route_obj
                                    #     alloc_table[(t,beta, c, d, i, S)] = current_alloc
                                    #     route_table[(t, beta, c, d, i, S)] = [j]
                                    # elif current_route_obj == dp_table[(t, beta, c, d, i, S)]:
                                    #     if j not in route_table[(t, beta, c, d, i, S)]:
                                    #         route_table[(t, beta, c, d, i, S)].append(j)
                                print(f'The objective at state {(t, beta, c, d, i, S)} is {dp_table[(t, beta, c, d, i, S)]}, with the optimal allocation {alloc_table[(t, beta, c, d, i, S)]} and next node {route_table[(t, beta, c, d, i, S)]}.')
    return dp_table, alloc_table, route_table

def count_decimals(number):
    s = str(number)
    if '.' in s:
        return len(s.split('.')[1].rstrip('0'))
    return 0

if __name__ == "__main__":
    D1 = {1: 1 / 5, 2: 1 / 5, 3: 1 / 5, 4: 1 / 5, 5: 1 / 5}
    D2 = {1: 1 / 10, 2: 1 / 5, 3: 2 / 5, 4: 1 / 5, 5: 1 / 10}
    D3 = {1: 2 / 5, 2: 3 / 10, 3: 1 / 5, 4: 7 / 100, 5: 3 / 100}
    D4 = {1: 3 / 100, 2: 7 / 100, 3: 1 / 5, 4: 3 / 10, 5: 2 / 5}
    D5 = {1: 2 / 5, 2: 3 / 40, 3: 1 / 20, 4: 3 / 40, 5: 2 / 5}
    D6 = {1: 3 / 10, 2: 2 / 5, 3: 1 / 5, 4: 7 / 100, 5: 3 / 100}
    D7 = {1: 1 / 5, 2: 7 / 100, 3: 3 / 100, 4: 3 / 10, 5: 2 / 5}
    D8 = {1: 2 / 5, 2: 3 / 100, 3: 3 / 10, 4: 7 / 100, 5: 1 / 5}
    horizon = 3
    F = [D1, D2, D3]
    auto_step_size = 0.1
    beta_step_size = 0.01
    beta_grid = np.round(np.arange(0, 1 + 1e-10, beta_step_size), count_decimals(beta_step_size))
    capacity_grid = np.round(np.arange(0, 15 + 1e-10, auto_step_size), count_decimals(auto_step_size))
    solve_dynamic_manshadi_DP(F, horizon, beta_grid, capacity_grid, auto_step_size)
    print('Finish')
    # F = [D1, D2, D3, D4, D5, D6, D7, D8]
    # # algorithm = ["equating", "brute-force"]
    # for i in range(len(F)):
    #     D = F[i]
    #     demands = list(D.keys())
    #     d_max = max(demands)
    #     d_mean = np.mean(demands)
    #     horizon = 3
    #     auto_step_size = 0.1
    #     beta_step_size = 0.01
    #     beta_grid = np.round(np.arange(0, 1 + 1e-10, beta_step_size), count_decimals(beta_step_size))
    #     # step_size = 0.1
    #     # auto_step_size = min(step_size, cal_gcd(demands))
    #     capacity_list = np.arange(0.1, (d_max / d_mean) + 0.1, 0.1)
    #     c_max = np.max(capacity_list) * d_mean * horizon
    #     capacity_grid = np.round(np.arange(0, c_max + 1e-10, auto_step_size), count_decimals(auto_step_size))
    #     # brute-force search
    #     try:
    #         dp_table_brute_force = {}
    #         alloc_table_brute_force = {}
    #         with open(f'./manshadi_dp_table/revised_discrete_type_{i}.csv', mode='r', newline='') as file:
    #             # with open(f'./manshadi_dp_table/discrete_type_{i}_shift_{distance}.csv', mode='r', newline='') as file:
    #             reader = csv.DictReader(file)
    #             for row in reader:
    #                 state = ast.literal_eval(row["state"])
    #                 dp_table_brute_force[state] = float(row["value"])
    #                 alloc_table_brute_force[state] = float(row["allocation"])
    #     except FileNotFoundError:
    #         dp_results = {"state": [], "value": [], "allocation": []}
    #         dp_table_brute_force, alloc_table_brute_force = revised_end_solve_homogeneous_manshadi_DP(D, horizon, beta_grid, capacity_grid,
    #                                                                           auto_step_size)
    #         for state in dp_table_brute_force.keys():
    #             dp_results["state"].append(state)
    #             dp_results["value"].append(dp_table_brute_force[state])
    #             dp_results["allocation"].append(alloc_table_brute_force[state])
    #         df = pd.DataFrame(dp_results)
    #         df.to_csv(f'./manshadi_dp_table/revised_discrete_type_{i}.csv', index=False)
    #     # reduced search space
    #     try:
    #         dp_table_equate = {}
    #         alloc_table_equate = {}
    #         with open(f'./manshadi_dp_table/revised_euqate_discrete_type_{i}.csv', mode='r', newline='') as file:
    #             # with open(f'./manshadi_dp_table/discrete_type_{i}_shift_{distance}.csv', mode='r', newline='') as file:
    #             reader = csv.DictReader(file)
    #             for row in reader:
    #                 state = ast.literal_eval(row["state"])
    #                 dp_table_equate[state] = float(row["value"])
    #                 alloc_table_equate[state] = float(row["allocation"])
    #     except FileNotFoundError:
    #         dp_results = {"state": [], "value": [], "allocation": []}
    #         dp_table_equate, alloc_table_equate = revised_equating_solve_homogeneous_manshadi_DP(D, horizon, beta_grid, capacity_grid,
    #                                                                           auto_step_size)
    #         for state in dp_table_equate.keys():
    #             dp_results["state"].append(state)
    #             dp_results["value"].append(dp_table_equate[state])
    #             dp_results["allocation"].append(alloc_table_equate[state])
    #         df = pd.DataFrame(dp_results)
    #         df.to_csv(f'./manshadi_dp_table/revised_equate_discrete_type_{i}.csv', index=False)
    #     print(f'distribution type {i}: {alloc_table_brute_force == alloc_table_equate}')
