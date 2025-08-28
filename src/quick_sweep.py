import os
import random

import pandas as pd

from src.annealing import simulated_annealing
from src.formulation import k_shortest_candidates
from src.network_builder import build_network


def build_random_demand(G, n=25, seed=7):
    rnd = random.Random(seed)
    nodes = list(G.nodes)
    return [tuple(rnd.sample(nodes, 2)) for _ in range(n)]

def final_violations(G, candidate_lists, state):
    # map state -> paths
    paths = [candidate_lists[i][state[i]] for i in range(len(state))]
    # compute violations (same logic as your compute_capacity_violation)
    usage = {tuple(sorted(e)):0 for e in G.edges}
    for p in paths:
        for u, v in zip(p[:-1], p[1:]):
            usage[tuple(sorted((u,v)))] += 1
    v = 0
    for (u, vtx), load in usage.items():
        cap = G[u][vtx]["capacity"]
        if load > cap:
            v += load - cap
    return v

def run_once(penalty):
    G = build_network()
    demand = build_random_demand(G, n=25, seed=7)
    cands = k_shortest_candidates(G, demand, k=4)
    state, best_cost, _loads = simulated_annealing(
        G, cands, episodes=60, temp_start=50.0, temp_end=0.5,
        congestion_penalty_coef=penalty, log_csv=f"sa_log_pen{penalty}.csv", seed=123
    )
    v = final_violations(G, cands, state)
    return {"penalty": penalty, "best_cost": best_cost, "violations": v}

if __name__ == "__main__":
    rows = []
    for p in [1.0, 5.0, 10.0, 25.0, 50.0]:
        rows.append(run_once(p))
    df = pd.DataFrame(rows)
    print(df.sort_values("penalty"))
    df.to_csv("quick_sweep_summary.csv", index=False)