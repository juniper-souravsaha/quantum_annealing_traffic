import random
import networkx as nx
from .annealing import (
    compute_edge_loads_from_state,
    objective_cost,
    compute_capacity_violation,
)

def shortest_path_baseline(G, demands, candidate_lists, congestion_penalty_coef=10.0):
    """
    Each demand takes the shortest path.
    """
    state = []
    for i, (src, dst, demand) in enumerate(demands):
        # candidate_lists[i] already contains k-shortest paths
        shortest_idx = min(
            range(len(candidate_lists[i])),
            key=lambda j: sum(G[u][v]["weight"] for u, v in zip(candidate_lists[i][j][:-1], candidate_lists[i][j][1:]))
        )
        state.append(shortest_idx)

    paths = [candidate_lists[i][state[i]] for i in range(len(state))]
    cost = objective_cost(G, candidate_lists, state, congestion_penalty_coef)
    loads = compute_edge_loads_from_state(G, candidate_lists, state)
    violations = compute_capacity_violation(G, paths)

    return state, cost, loads, violations


def random_routing_baseline(G, demands, candidate_lists, congestion_penalty_coef=10.0, seed=123):
    """
    Each demand randomly picks one of its candidate paths.
    """
    rnd = random.Random(seed)
    state = [rnd.randrange(len(cands)) for cands in candidate_lists]

    paths = [candidate_lists[i][state[i]] for i in range(len(state))]
    cost = objective_cost(G, candidate_lists, state, congestion_penalty_coef)
    loads = compute_edge_loads_from_state(G, candidate_lists, state)
    violations = compute_capacity_violation(G, paths)

    return state, cost, loads, violations