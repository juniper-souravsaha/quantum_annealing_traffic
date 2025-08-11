import argparse
from src import network, traffic_sim, annealing, visualize
from src.network_builder import build_network
import networkx as nx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--temp_start', type=float, default=100.0)
    parser.add_argument('--temp_end', type=float, default=1.0)
    args = parser.parse_args()

    G = network.create_grid_network(4, 4)
    demand = traffic_sim.generate_demand(G, num_vehicles=20)

    best_paths, best_cost = annealing.simulated_annealing(
        G, demand, args.temp_start, args.temp_end, args.episodes
    )

    print(f"Final best cost: {best_cost:.2f}")
    visualize.plot_network(G, best_paths)
    
    # # Build road network
    # G = build_network()

    # print("Road Network Nodes:", list(G.nodes))
    # print("Road Network Edges (with attributes):")
    # for u, v, data in G.edges(data=True):
    #     print(f"{u} ↔ {v} | Time: {data['time']} min | Capacity: {data['capacity']} vehicles")

    # # Example: shortest path by travel time
    # path = nx.shortest_path(G, source="A", target="D", weight="time")
    # travel_time = nx.shortest_path_length(G, source="A", target="D", weight="time")
    # print(f"\nShortest path A → D: {path} (Total time: {travel_time} min)")


if __name__ == "__main__":
    main()