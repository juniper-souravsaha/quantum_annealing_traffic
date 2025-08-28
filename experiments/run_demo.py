import argparse

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from src.annealing import compute_capacity_violation, simulated_annealing
from src.baselines import shortest_path_baseline
from src.graph_setup import (build_large_graph, enumerate_candidate_paths,
                             generate_demands)


def run_demo(grid_size=6, num_demands=50, episodes=150, penalty=10.0, k_paths=3, seed=42):
    # 1. Build network
    G = build_large_graph(grid_size=grid_size, seed=seed)
    print(f"Built {grid_size}x{grid_size} grid with {len(G.nodes)} nodes and {len(G.edges)} edges")

    # 2. Generate demands
    demands = generate_demands(G, num_demands=num_demands, seed=seed)
    print(f"Generated {len(demands)} demands")

    # 3. Candidate paths
    candidate_lists = enumerate_candidate_paths(G, demands, k=k_paths)

    # 4. Baseline: shortest-path allocation
    sp_state, sp_cost, _, sp_viol = shortest_path_baseline(G, demands, candidate_lists)
    print(f"[Baseline] Cost={sp_cost:.2f}, Violations={sp_viol}")

    # 5. Simulated Annealing run
    log_file = f"results/logs/demo_sa_grid{grid_size}_d{num_demands}.csv"
    best_state, best_cost, final_loads = simulated_annealing(
        G,
        candidate_lists,
        episodes=episodes,
        congestion_penalty_coef=penalty,
        log_csv=log_file,
        seed=seed,
    )
    chosen_paths = [candidate_lists[i][best_state[i]] for i in range(len(best_state))]
    violations = compute_capacity_violation(G, chosen_paths)
    print(f"[Simulated Annealing] Cost={best_cost:.2f}, Violations={violations}")

    # 6. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Load distribution
    loads = nx.get_edge_attributes(G, "load")
    pos = nx.spring_layout(G, seed=seed)
    nx.draw(G, pos, ax=ax[0], with_labels=True, node_size=300)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=loads, ax=ax[0])
    ax[0].set_title("Final Edge Loads (SA Allocation)")

    # Cost/Violation over time
    df = pd.read_csv(log_file)
    ax[1].plot(df["episode"], df["best_cost"], label="BestCost")
    ax[1].plot(df["episode"], df["violations"], label="Violations")
    ax[1].legend()
    ax[1].set_title("SA Optimization Progress")
    ax[1].set_xlabel("Episode")

    plt.tight_layout()
    out_img = f"results/plots/demo_grid{grid_size}_d{num_demands}.png"
    plt.savefig(out_img)
    plt.show()

    return {
        "baseline_cost": sp_cost,
        "baseline_viol": sp_viol,
        "sa_cost": best_cost,
        "sa_viol": violations,
        "log_file": log_file,
        "plot_file": out_img,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Traffic Allocation Demo with Simulated Annealing")
    parser.add_argument("--grid", type=int, default=6, help="Grid size (NxN)")
    parser.add_argument("--demands", type=int, default=50, help="Number of traffic demands")
    parser.add_argument("--episodes", type=int, default=150, help="Number of annealing episodes")
    parser.add_argument("--penalty", type=float, default=10.0, help="Congestion penalty coefficient")
    parser.add_argument("--k_paths", type=int, default=3, help="Number of candidate paths per demand")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    results = run_demo(
        grid_size=args.grid,
        num_demands=args.demands,
        episodes=args.episodes,
        penalty=args.penalty,
        k_paths=args.k_paths,
        seed=args.seed,
    )

    print("\n=== Final Results ===")
    print(results)