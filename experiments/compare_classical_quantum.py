from src.annealing import simulated_annealing
from src.graph_setup import enumerate_candidate_paths, generate_network
from src.quantum_solvers import solve_dwave, solve_qaoa
from src.qubo_formulation import build_qubo


def compare_experiment():
    # Build problem
    G, demands = generate_network(n=10, edge_prob=0.3, capacity=10)
    candidate_lists = enumerate_candidate_paths(G, demands, k=3)

    # Classical simulated annealing
    sa_state, sa_cost, _, sa_viol = simulated_annealing(G, demands, candidate_lists)
    print("Classical SA:", sa_cost, sa_viol)

    # Quantum: QUBO build
    bqm = build_qubo(G, demands, candidate_lists)

    # Try D-Wave (requires credentials)
    try:
        sol, energy = solve_dwave(bqm)
        print("Quantum D-Wave:", energy, sol)
    except Exception as e:
        print("D-Wave not available:", e)

    # Try QAOA
    try:
        sol, val = solve_qaoa(bqm)
        print("Quantum QAOA:", val, sol)
    except Exception as e:
        print("QAOA failed:", e)

if __name__ == "__main__":
    compare_experiment()