import itertools
import pandas as pd
import matplotlib.pyplot as plt

from src.graph_setup import build_toy_graph, generate_demands, enumerate_candidate_paths
from src.annealing import simulated_annealing, compute_capacity_violation
from src.baselines import shortest_path_baseline, random_routing_baseline


def run_extended_sweep(
    demand_sizes=[25, 50, 75, 100],
    penalties=[1.0, 5.0, 10.0, 25.0, 50.0],
    episodes=100,
    output_csv="extended_sweep_results.csv",
):
    results = []

    for dsize, pen in itertools.product(demand_sizes, penalties):
        print(f"=== Running SA with demand_size={dsize}, penalty={pen} ===")
        G = build_toy_graph()
        demands = generate_demands(G, num_demands=dsize, seed=42)
        candidate_lists = enumerate_candidate_paths(G, demands, k=3)

        best_state, best_cost, final_loads = simulated_annealing(
            G,
            candidate_lists,
            episodes=episodes,
            congestion_penalty_coef=pen,
            log_csv=f"results/logs/grid_dsize{dsize}_pen{pen}.csv",
            seed=42,
        )

        chosen_paths = [candidate_lists[i][best_state[i]] for i in range(len(best_state))]
        violations = compute_capacity_violation(G, chosen_paths)

        results.append({
            "method": "simulated_annealing",
            "demand_size": dsize,
            "penalty": pen,
            "best_cost": best_cost,
            "violations": violations,
        })

    # === Baselines for each demand size ===
    for dsize in demand_sizes:
        G = build_toy_graph()
        demands = generate_demands(G, num_demands=dsize, seed=42)
        candidate_lists = enumerate_candidate_paths(G, demands, k=3)

        # Shortest path
        sp_state, sp_cost, _, sp_viol = shortest_path_baseline(G, demands, candidate_lists)
        results.append({
            "method": "shortest_path",
            "demand_size": dsize,
            "penalty": None,
            "best_cost": sp_cost,
            "violations": sp_viol,
        })

        # Random routing
        rand_state, rand_cost, _, rand_viol = random_routing_baseline(G, demands, candidate_lists)
        results.append({
            "method": "random_routing",
            "demand_size": dsize,
            "penalty": None,
            "best_cost": rand_cost,
            "violations": rand_viol,
        })

    df = pd.DataFrame(results)

    # Save full results to CSV
    df.to_csv("results/logs/"+output_csv, index=False)
    print(f"\n[Done] Extended sweep results saved to {output_csv}")
    return df


def plot_heatmaps(df):
    """Plot heatmaps only for simulated annealing (ignores baselines)."""
    sa_df = df[df["method"] == "simulated_annealing"]

    # Pivot for heatmap
    cost_matrix = sa_df.pivot(index="demand_size", columns="penalty", values="best_cost")
    viol_matrix = sa_df.pivot(index="demand_size", columns="penalty", values="violations")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    c1 = axes[0].imshow(cost_matrix, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(cost_matrix.columns)))
    axes[0].set_xticklabels(cost_matrix.columns)
    axes[0].set_yticks(range(len(cost_matrix.index)))
    axes[0].set_yticklabels(cost_matrix.index)
    axes[0].set_title("Best Cost (SA only)")
    fig.colorbar(c1, ax=axes[0])

    c2 = axes[1].imshow(viol_matrix, cmap="magma", aspect="auto")
    axes[1].set_xticks(range(len(viol_matrix.columns)))
    axes[1].set_xticklabels(viol_matrix.columns)
    axes[1].set_yticks(range(len(viol_matrix.index)))
    axes[1].set_yticklabels(viol_matrix.index)
    axes[1].set_title("Violations (SA only)")
    fig.colorbar(c2, ax=axes[1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = run_extended_sweep()
    print("\nSweep results:\n", df)

    plot_heatmaps(df)