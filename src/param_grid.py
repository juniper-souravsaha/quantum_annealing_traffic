import random
from itertools import product

import pandas as pd

from src.annealing import simulated_annealing
from src.formulation import k_shortest_candidates
from src.network_builder import build_network


def build_random_demand(G, n=60, seed=11):
    rnd = random.Random(seed)
    nodes = list(G.nodes)
    return [tuple(rnd.sample(nodes, 2)) for _ in range(n)]

def final_violations(G, cands, state):
    usage = {tuple(sorted(e)):0 for e in G.edges}
    paths = [cands[i][state[i]] for i in range(len(state))]
    for p in paths:
        for u, v in zip(p[:-1], p[1:]):
            usage[tuple(sorted((u,v)))] += 1
    v = 0
    for (u, w), load in usage.items():
        cap = G[u][w]["capacity"]
        if load > cap:
            v += load - cap
    return v

if __name__ == "__main__":
    G = build_network()
    demand = build_random_demand(G, n=60, seed=11)
    cands = k_shortest_candidates(G, demand, k=4)

    rows = []
    for ep, mpe, pen in product([60, 120, 200], [len(cands)//4, len(cands)//2, len(cands)], [5.0, 10.0, 25.0]):
        state, best_cost, _ = simulated_annealing(
            G, cands, episodes=ep, temp_start=50, temp_end=0.5,
            moves_per_episode=mpe, congestion_penalty_coef=pen,
            log_csv=f"grid_ep{ep}_m{mpe}_pen{pen}.csv", seed=123
        )
        v = final_violations(G, cands, state)
        rows.append({"episodes": ep, "moves_per_episode": mpe, "penalty": pen,
                     "best_cost": best_cost, "violations": v})
    df = pd.DataFrame(rows).sort_values(["violations", "best_cost"])
    print(df)
    df.to_csv("param_grid_summary.csv", index=False)