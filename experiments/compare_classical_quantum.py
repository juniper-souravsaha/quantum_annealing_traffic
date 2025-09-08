from src.annealing import compute_capacity_violation, simulated_annealing
from src.graph_setup import build_large_graph, enumerate_candidate_paths, generate_demands
from src.quantum_solvers import solve_dwave, solve_qaoa
from src.qubo_formulation import build_qubo


def compare_experiment(grid_size=4, num_demands=10, episodes=50, penalty=10.0, k_paths=3, seed=42):
    # 1. Build network
    G = build_large_graph(grid_size=grid_size, seed=seed)
    print(f"Built {grid_size}x{grid_size} grid with {len(G.nodes)} nodes and {len(G.edges)} edges")

    # 2. Generate demands
    demands = generate_demands(G, num_demands=num_demands, seed=seed)
    print(f"Generated {len(demands)} demands")

    # 3. Candidate paths
    candidate_lists = enumerate_candidate_paths(G, demands, k=k_paths)

    # 4. Simulated Annealing run
    log_file = f"results/logs/demo_sa_grid{grid_size}_d{num_demands}.csv"
    sa_state, sa_cost, _ = simulated_annealing(
        G,
        candidate_lists,
        episodes=episodes,
        congestion_penalty_coef=penalty,
        log_csv=log_file,
        seed=seed,
    )
    chosen_paths = [candidate_lists[i][sa_state[i]] for i in range(len(sa_state))]
    sa_viol = compute_capacity_violation(G, chosen_paths)
    print("Classical SA:", sa_cost, sa_viol)

    # Quantum: QUBO build
    bqm = build_qubo(G, demands, candidate_lists)

    # sol, energy = solve_dwave(bqm)
    # print("Quantum D-Wave:", energy, sol)

    # Try QAOA
    sol, val = solve_qaoa(bqm, reps=1, maxiter=20, optimizer_name="COBYLA")
    print("Quantum QAOA:", val, sol)

if __name__ == "__main__":
    compare_experiment()