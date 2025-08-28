import random
import networkx as nx

from src.network_builder import build_network
from src.formulation import k_shortest_candidates
from src.annealing import simulated_annealing
from src import visualize

def build_random_demand(G, n=25, seed=7):
    rnd = random.Random(seed)
    nodes = list(G.nodes)
    demand = []
    for _ in range(n):
        s, t = rnd.sample(nodes, 2)
        demand.append((s, t))
    return demand

def main():
    # 1) Build the capacity-aware network
    G = build_network()
    print("Nodes:", list(G.nodes))
    print("Edges with (time, capacity):")
    for u, v, d in G.edges(data=True):
        print(f"  {u} - {v} | time={d.get('time')} | capacity={d.get('capacity')}")

    # 2) Quick shortest path check (sanity)
    try:
        path = nx.shortest_path(G, source="A", target="D", weight="time")
        t = nx.shortest_path_length(G, source="A", target="D", weight="time")
        print(f"\nSanity shortest path A→D: {path} (time={t})")
    except Exception as e:
        print("Shortest path sanity check failed:", e)

    # 3) Build random OD demand and candidate paths
    demand = build_random_demand(G, n=25, seed=7)
    print(f"\nDemand size: {len(demand)}")
    candidate_lists = k_shortest_candidates(G, demand, k=4)  # 4 candidates per vehicle

    # 4) Run SA on the state space
    best_state, best_cost, final_loads = simulated_annealing(
        G,
        candidate_lists,
        episodes=80,
        temp_start=50.0,
        temp_end=0.5,
        moves_per_episode=None,           # defaults to ~N/2
        congestion_penalty_coef=10.0,     # tune later
        log_csv="sa_log.csv",
        seed=123,
    )

    # 5) Convert best_state -> list of paths for plotting
    best_paths = [candidate_lists[i][best_state[i]] for i in range(len(candidate_lists))]
    print(f"\nFinal best cost: {best_cost:.2f}")
    visualize.plot_network(G, best_paths)

if __name__ == "__main__":
    main()