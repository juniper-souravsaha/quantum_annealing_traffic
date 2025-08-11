import random
import networkx as nx

def generate_demand(G, num_vehicles=20):
    nodes = list(G.nodes())
    demand = []
    for _ in range(num_vehicles):
        start, end = random.sample(nodes, 2)
        demand.append((start, end))
    return demand

def assign_paths_randomly(G, demand):
    paths = []
    for start, end in demand:
        path = nx.shortest_path(G, start, end, weight='length')
        paths.append(path)
    return paths

def compute_edge_loads(G, paths):
    loads = {edge: 0 for edge in G.edges()}
    for path in paths:
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if (u, v) in loads:
                loads[(u, v)] += 1
            elif (v, u) in loads:
                loads[(v, u)] += 1
    return loads