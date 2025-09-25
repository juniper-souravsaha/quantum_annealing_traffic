import logging
import time
from graph_setup import build_large_graph, enumerate_candidate_paths, generate_demands
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qubo_formulation import build_qubo
from src.quantum_solvers import solve_sa, solve_qaoa

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("experiment")


# ---------------------------
# Experiment Runner
# ---------------------------
def run_experiment():
    sizes = [3, 4, 5]  # keep small for demo
    sa_times, qaoa_times = [], []
    sa_costs, qaoa_costs = [], []
    sa_solutions, qaoa_solutions = [], []
    seed = 42
    k_paths = 3  # candidate paths per demand

    for n in sizes:
        logger.info(f"\n=== Running for network size {n} ===")
        G = build_large_graph(grid_size=n, seed=seed)
        logger.info(f"Built {n}x{n} grid with {len(G.nodes)} nodes and {len(G.edges)} edges")

        demands = generate_demands(G, num_demands=4, seed=seed)
        logger.info(f"Generated {len(demands)} demands")

        candidate_lists = enumerate_candidate_paths(G, demands, k=k_paths)
        bqm = build_qubo(G, demands, candidate_lists)

        # ---------------------------
        # Classical SA
        # ---------------------------
        t0 = time.time()
        sa_sol, sa_val = solve_sa(bqm)
        sa_times.append(time.time() - t0)
        sa_costs.append(sa_val)
        sa_solutions.append(sa_sol)
        logger.info(f"SA: value={sa_val:.2f}, time={sa_times[-1]:.2f}s")
        log_solution_summary(sa_sol, "SA")

        # ---------------------------
        # Quantum QAOA
        # ---------------------------
        try:
            t0 = time.time()
            q_sol, q_val = solve_qaoa(bqm, reps=1, maxiter=20, optimizer_name="COBYLA")
            qaoa_times.append(time.time() - t0)
            qaoa_costs.append(q_val)
            qaoa_solutions.append(q_sol)
            logger.info(f"QAOA: value={q_val:.2f}, time={qaoa_times[-1]:.2f}s")
            log_solution_summary(q_sol, "QAOA")
        except Exception as e:
            logger.error(f"QAOA failed: {e}")
            qaoa_times.append(None)
            qaoa_costs.append(np.nan)

    # ---------------------------
    # Summary Across Sizes
    # Plot Solutions
    # ---------------------------
    logger.info("\n=== Experiment Summary ===")
    for i, n in enumerate(sizes):
        logger.info(f"Network size {n}: "
                    f"SA value={sa_costs[i]:.2f}, SA time={sa_times[i]:.2f}s | "
                    f"QAOA value={qaoa_costs[i]:.2f}, QAOA time={qaoa_times[i]}")
        plot_solution(G, sa_solutions[i], title=f"Classical SA (val={sa_costs[i]:.2f})")
        if qaoa_costs[-1] is not np.nan:
            plot_solution(G, qaoa_solutions[i], title=f"Quantum QAOA (val={qaoa_costs[i]:.2f})")
    logger.info("=== End of Experiment ===")

    # ---------------------------
    # Plot Results
    # ---------------------------
    plot_results(sizes, sa_costs, qaoa_costs, sa_times, qaoa_times)


# ---------------------------
# Logging Helpers
# ---------------------------
def log_solution_summary(solution, tag="Solver"):
    """Summarize chosen paths and nodes from solution dictionary."""
    chosen = [var for var, val in solution.items() if val == 1]
    logger.info(f"[{tag}] Selected {len(chosen)} variables → {chosen[:10]}{'...' if len(chosen) > 10 else ''}")

    # Optionally count per-demand selections if variables are x_i_j
    demand_counts = {}
    for var, val in solution.items():
        if val == 1 and var.startswith("x_"):
            try:
                _, i, j = var.split("_")
                demand_counts[int(i)] = demand_counts.get(int(i), 0) + 1
            except ValueError:
                continue
    logger.info(f"[{tag}] Demand path counts: {demand_counts}")


# ---------------------------
# Plotting Helpers
# ---------------------------
def plot_solution(G, solution, title="Solution"):
    """Highlight nodes that are used in selected paths."""
    pos = nx.spring_layout(G, seed=42)

    chosen_nodes = set()
    for var, val in solution.items():
        if val == 1 and var.startswith("x_"):
            try:
                _, i, j = var.split("_")
                chosen_nodes.add(int(i))  # mark demand source
            except ValueError:
                pass  # ignore variables not following x_i_j format

    color_map = ["lightgreen" if node in chosen_nodes else "lightgrey" for node in G.nodes()]

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=600, edgecolors="black")
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_results(sizes, sa_costs, qaoa_costs, sa_times, qaoa_times):
    """Compare costs and runtimes."""
    plt.figure(figsize=(10, 4))

    # Cost comparison
    plt.subplot(1, 2, 1)
    plt.plot(sizes, sa_costs, "o-", label="SA cost")
    if not all(np.isnan(c) for c in qaoa_costs):
        plt.plot(sizes, qaoa_costs, "s-", label="QAOA cost")
    plt.xlabel("Network size (nodes)")
    plt.ylabel("Solution cost")
    plt.title("Cost Comparison")
    plt.legend()

    # Runtime comparison
    plt.subplot(1, 2, 2)
    plt.plot(sizes, sa_times, "o-", label="SA time")
    if any(t is not None for t in qaoa_times):
        plt.plot(sizes, [t if t is not None else 0 for t in qaoa_times], "s-", label="QAOA time")
    plt.xlabel("Network size (nodes)")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    run_experiment()