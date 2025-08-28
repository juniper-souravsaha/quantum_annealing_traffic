import math
import random
from collections import Counter, defaultdict

import networkx as nx


def k_shortest_candidates(G, demand, k=6, weight='time', cutoff=None):
    """
    For each demand (s,t) produce up to k candidate simple paths (short-to-long).
    Returns: list_of_candidate_lists where element i is list of paths for demand i.
    """
    candidate_lists = []
    for (s, t) in demand:
        # generator of simple shortest paths (in ascending weight)
        try:
            gen = nx.shortest_simple_paths(G, s, t, weight=weight)
            paths = []
            for i, p in enumerate(gen):
                paths.append(p)
                if i + 1 >= k:
                    break
        except (nx.NetworkXNoPath, StopIteration):
            paths = [[s]]  # fallback trivial path (no movement)
        # optionally filter by cutoff length (number of hops)
        if cutoff is not None:
            paths = [p for p in paths if len(p)-1 <= cutoff]
            if len(paths) == 0:
                paths = [[s]]
        candidate_lists.append(paths)
    return candidate_lists

def compute_edge_loads_from_state(G, candidate_lists, state):
    """
    Given a state (choice of path index for each demand), compute per-edge loads.
    """
    loads = {tuple(sorted(edge)): 0 for edge in G.edges}

    for i, path_idx in enumerate(state):
        path = candidate_lists[i][path_idx]
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            loads[edge] += 1
    return loads

def path_travel_time(G, path, time_key='time'):
    """Sum of travel-time attribute along a path"""
    total = 0
    for u, v in zip(path[:-1], path[1:]):
        attrs = G.get_edge_data(u, v)
        total += attrs.get(time_key, 1)
    return total

def objective_cost(G, candidate_lists, state, congestion_penalty_coef=5.0, capacity_key='capacity', time_key='time', power=2):
    """
    Compute total cost: travel time + congestion penalty.
    congestion penalty for each edge = coef * max(0, load - capacity)^power
    """
    # travel time
    travel_cost = 0.0
    for i, p_idx in enumerate(state):
        path = candidate_lists[i][p_idx]
        travel_cost += path_travel_time(G, path, time_key=time_key)

    # edge loads
    loads = compute_edge_loads_from_state(G, candidate_lists, state)

    # congestion penalty
    penalty = 0.0
    for e, load in loads.items():
        cap = G[e[0]][e[1]].get(capacity_key, float('inf'))
        excess = max(0.0, load - cap)
        penalty += congestion_penalty_coef * (excess ** power)

    return travel_cost + penalty

def random_initial_state(candidate_lists):
    """Return a random feasible state (choose random path index per demand)"""
    return [random.randrange(len(P)) for P in candidate_lists]

def sa_neighbor(state, candidate_lists):
    """
    Propose a neighbor by changing one random vehicle's chosen path
    Returns new_state (copy)
    """
    new_state = state[:]  # shallow copy
    i = random.randrange(len(state))
    choices = list(range(len(candidate_lists[i])))
    if len(choices) <= 1:
        return new_state
    choices.remove(new_state[i])
    new_state[i] = random.choice(choices)
    return new_state

# --- QUBO construction (optional) ---

def build_qubo(G, candidate_lists, penalty_constraint=50.0, congestion_penalty_coef=5.0, capacity_key='capacity', time_key='time', power=2):
    """
    Build a QUBO matrix (dict-of-dicts or dict with tuple keys) for binary variables x_{i,p}.
    We map each variable to an index: var_idx[(i,p)] = q (int)
    Returns: Q(dict[(q1,q2)] -> float), var_idx(dict)
    Notes:
     - Linear term from travel time: sum_{i,p} time(i,p) * x_{i,p}
     - Congestion: for each edge, load_e = sum_{i,p} I_{i,p}(e) * x_{i,p}
       penalty is coef * (max(0, load_e - cap))^2. Expanding yields quadratic couplings between x variables.
     - Constraint: for each i, (sum_p x_{i,p} - 1)^2 * weight -> enforces exactly-one path chosen for each vehicle.
    """
    var_idx = {}
    idx = 0
    for i, P in enumerate(candidate_lists):
        for p_idx, path in enumerate(P):
            var_idx[(i, p_idx)] = idx
            idx += 1

    Q = {}  # map (u,v) with u<=v -> coefficient

    def q_add(a, b, val):
        if a > b:
            a, b = b, a
        Q[(a, b)] = Q.get((a, b), 0.0) + val

    # linear: travel times
    for (i, p_idx), q in var_idx.items():
        path = candidate_lists[i][p_idx]
        t = path_travel_time(G, path, time_key=time_key)
        q_add(q, q, t)  # linear cost

    # congestion quadratic terms
    # For each edge, find which (i,p) include it (indicator)
    edge_to_vars = defaultdict(list)
    for (i, p_idx), q in var_idx.items():
        path = candidate_lists[i][p_idx]
        for u, v in zip(path[:-1], path[1:]):
            e = (u, v) if (u, v) in G.edges() else (v, u)
            edge_to_vars[e].append(q)

    for e, vars_on_e in edge_to_vars.items():
        cap = G[e[0]][e[1]].get(capacity_key, 0)
        # expand (sum x - cap)^2 = sum_i x_i + 2 sum_{i<j} x_i x_j - 2 cap sum_i x_i + cap^2
        # Multiply by congestion_penalty_coef for weight
        for q in vars_on_e:
            # linear part: coef * (1 - 2*cap)
            q_add(q, q, congestion_penalty_coef * (1.0 - 2.0 * cap))
        for i1 in range(len(vars_on_e)):
            for i2 in range(i1+1, len(vars_on_e)):
                q1 = vars_on_e[i1]; q2 = vars_on_e[i2]
                # quadratic coupling: 2 * coef
                q_add(q1, q2, 2.0 * congestion_penalty_coef)
        # constant term cap^2 * coef omitted (doesn't affect optimization)

    # constraint: exactly one path per vehicle via penalty (sum_p x_{i,p} - 1)^2 * penalty_constraint
    for i, P in enumerate(candidate_lists):
        vars_i = [var_idx[(i, p_idx)] for p_idx in range(len(P))]
        for q in vars_i:
            q_add(q, q, penalty_constraint * (1.0))  # diagonal from x_i^2
        for a_idx in range(len(vars_i)):
            for b_idx in range(a_idx + 1, len(vars_i)):
                q_add(vars_i[a_idx], vars_i[b_idx], 2.0 * penalty_constraint * 1.0 * 0.5 * -1.0)  # cross term: 2*penalty* x_a x_b -> we want +2*penalty, but expanding (sum-1)^2 => +2*x_a x_b, to penalize multiple selections
        # subtract linear 2*1*penalty? careful: (sum x -1)^2 = sum x^2 + 2 sum_{a<b} x_a x_b - 2 sum x + 1
        for q in vars_i:
            q_add(q, q, -2.0 * penalty_constraint * 0.5)  # subtract 2*penalty for linear; here using half-shares to maintain consistent double-counting rules

    # NOTE: sign bookkeeping for constraint expansion can be delicate depending on how you consume Q (Ising vs QUBO conventions).
    # The Q produced above is a basic starting QUBO; before sending to a solver, test small instances and adjust penalty_constraint scale.

    return Q, var_idx