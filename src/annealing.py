import math
import random
import csv
from src.formulation import (
    compute_edge_loads_from_state,
    objective_cost,
    sa_neighbor,
    random_initial_state,
)

def simulated_annealing(
    G,
    candidate_lists,
    episodes=80,
    temp_start=50.0,
    temp_end=0.5,
    moves_per_episode=None,
    congestion_penalty_coef=10.0,
    log_csv="sa_log.csv",
    seed=123,
):
    """
    SA over a discrete 'state' where state[i] is the chosen path index for demand i.
    Logs per-episode metrics to CSV and prints progress.
    Returns: (best_state, best_cost, final_edge_loads)
    """
    # Reproducibility
    rnd = random.Random(seed)
    random.seed(seed)

    # Initial state
    state = random_initial_state(candidate_lists)
    current_cost = objective_cost(G, candidate_lists, state, congestion_penalty_coef)
    best_state = state[:]
    best_cost = current_cost
    best_paths = candidate_lists

    # SA schedule
    if moves_per_episode is None:
        moves_per_episode = max(1, len(candidate_lists) // 2)
    cooling = (temp_end / temp_start) ** (1.0 / max(1, episodes))
    temp = temp_start

    # CSV header
    with open(log_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "episode", "temp", "current_cost", "best_cost",
            "acceptance_rate", "violations"
        ])

    # Main loop
    for ep in range(episodes):
        accepts = 0
        trials = 0
        accepted = 0

        for _ in range(moves_per_episode):
            trials += 1
            cand = sa_neighbor(state, candidate_lists)
            cand_cost = objective_cost(G, candidate_lists, cand, congestion_penalty_coef)
            delta = cand_cost - current_cost

            if delta <= 0 or rnd.random() < math.exp(-delta / max(1e-9, temp)):
                state = cand
                current_cost = cand_cost
                accepts += 1
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_state = state[:]

        acc_rate = accepts / trials if trials else 0.0
        violations = compute_capacity_violation(G, [candidate_lists[i][state[i]] for i in range(len(state))])
        # Log
        with open(log_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, f"{temp:.6f}", f"{current_cost:.6f}", f"{best_cost:.6f}", f"{acc_rate:.4f}"])

        if ep % max(1, episodes // 10) == 0 or ep == episodes - 1:
            print(f"Episode {ep}: temp={temp:.3f}, current={current_cost:.2f}, best={best_cost:.2f}, accept_rate={acc_rate:.2f}, Violations={violations}")

        temp *= cooling

    final_loads = compute_edge_loads_from_state(G, candidate_lists, best_state)
    return best_state, best_cost, final_loads

def compute_capacity_violation(G, paths):
    """
    Returns total violation amount (sum of overloads across all edges).
    """
    usage = {tuple(sorted(edge)): 0 for edge in G.edges}

    for path in paths:
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))  # normalize
            usage[edge] += 1

    violation = 0
    for (u, v), load in usage.items():
        cap = G[u][v]["capacity"]
        if load > cap:
            violation += load - cap
    return violation

def compute_cost(G, paths, congestion_penalty_coef):
    """
    Total cost = sum of travel times + penalty * capacity violations.
    """
    usage = {tuple(sorted(edge)): 0 for edge in G.edges}

    total_time = 0
    for path in paths:
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            usage[edge] += 1
            total_time += G[u][v]["time"]

    violation = 0
    for (u, v), load in usage.items():
        cap = G[u][v]["capacity"]
        if load > cap:
            violation += load - cap

    return total_time + congestion_penalty_coef * violation

def random_neighbor(G, paths):
    """
    Generate a neighbor by changing one random trip to a new shortest path.
    """
    new_paths = paths.copy()
    idx = random.randrange(len(paths))
    s, t = new_paths[idx][0], new_paths[idx][-1]

    try:
        # Pick a different path randomly among k-shortest
        k_paths = list(nx.shortest_simple_paths(G, s, t, weight="weight"))
        if len(k_paths) > 1:
            alt = random.choice(k_paths[1:])  # avoid best path
            new_paths[idx] = alt
    except nx.NetworkXNoPath:
        pass

    return new_paths