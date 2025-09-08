import time
from graph_setup import build_large_graph, enumerate_candidate_paths, generate_demands
import matplotlib.pyplot as plt
import networkx as nx
import dimod
import random

import numpy as np
from qubo_formulation import build_qubo
from src.quantum_solvers import solve_sa, solve_qaoa

# ---------------------------
# Main experiment
# ---------------------------
def run_experiment():
    sizes = [4]  # keep small for demo
    sa_times, qaoa_times = [], []
    sa_costs, qaoa_costs = [], []
    seed = 42
    k_paths = 3  # candidate paths per demand
    for n in sizes:
        print(f"\n=== Running for network size {n} ===")
        # G = generate_network(n)
        G = build_large_graph(grid_size=n, seed=seed)
        print(f"Built {n}x{n} grid with {len(G.nodes)} nodes and {len(G.edges)} edges")

        demands = generate_demands(G, num_demands=5, seed=seed)
        print(f"Generated {len(demands)} demands")

        candidate_lists = enumerate_candidate_paths(G, demands, k=k_paths)
        
        bqm = build_qubo(G, demands, candidate_lists)

        # Classical SA
        t0 = time.time()
        sa_sol, sa_val = solve_sa(bqm)
        sa_times.append(time.time() - t0)
        sa_costs.append(sa_val)
        print(f"SA: value={sa_val:.2f}, time={sa_times[-1]:.2f}s")
        print("SA Solution:", sa_sol, "Value:", sa_val)
        
        # Quantum QAOA
        try:
            t0 = time.time()
            q_sol, q_val = solve_qaoa(bqm, reps=1)
            qaoa_times.append(time.time() - t0)
            qaoa_costs.append(q_val)
            print(f"QAOA: value={q_val:.2f}, time={qaoa_times[-1]:.2f}s")
            print("QAOA Solution:", q_sol, "Value:", q_val)
        except Exception as e:
            print(f"QAOA failed: {e}")
            qaoa_times.append(None)
            qaoa_costs.append(np.nan)
        plot_solution(G, sa_sol, title=f"Classical SA Solution (val={sa_val:.2f})")
        plot_solution(G, q_sol, title=f"Quantum QAOA Solution (val={q_val:.2f})")


#     # ---------------------------
#     # Plot Results
#     # ---------------------------
#     plt.figure(figsize=(10, 4))

#     # Cost comparison
#     plt.subplot(1, 2, 1)
#     plt.plot(sizes, sa_costs, "o-", label="SA cost")
#     plt.plot(sizes, qaoa_costs, "s-", label="QAOA cost")
#     plt.xlabel("Network size (nodes)")
#     plt.ylabel("Solution cost")
#     plt.title("Cost Comparison")
#     plt.legend()

#     # Runtime comparison
#     plt.subplot(1, 2, 2)
#     plt.plot(sizes, sa_times, "o-", label="SA time")
#     plt.plot(sizes, qaoa_times, "s-", label="QAOA time")
#     plt.xlabel("Network size (nodes)")
#     plt.ylabel("Runtime (s)")
#     plt.title("Runtime Comparison")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

def plot_solution(G, solution, title="Solution"):
    pos = nx.spring_layout(G, seed=42)  # consistent layout
    
    # safely get bit assignment (default = 0 if missing)
    color_map = [
        'lightgreen' if solution.get(int(node), 0) == 0 else 'salmon'
        for node in G.nodes()
    ]
    
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=600, edgecolors="black")
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
    
    # edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    run_experiment()