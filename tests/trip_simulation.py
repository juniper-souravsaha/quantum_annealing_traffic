import random
import networkx as nx
from src.network_builder import build_network

def simulate_trips(num_trips=5):
    """
    Simulate a small number of trips and see which roads get used.
    Returns a dict: { (u, v): vehicles_count }
    """
    G = build_network()
    road_usage = {tuple(sorted((u, v))): 0 for u, v in G.edges}

    nodes = list(G.nodes)

    for _ in range(num_trips):
        start, end = random.sample(nodes, 2)
        path = nx.shortest_path(G, source=start, target=end, weight="time")

        # Count vehicles on each road used in this trip
        for i in range(len(path) - 1):
            edge = tuple(sorted((path[i], path[i + 1])))
            road_usage[edge] += 1

    return road_usage

if __name__ == "__main__":
    usage = simulate_trips(10)
    print("Road usage after 10 trips:")
    for road, count in usage.items():
        print(f"{road}: {count} vehicles")