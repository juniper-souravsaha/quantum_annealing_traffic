import networkx as nx
import random


def build_toy_graph():
    """Builds a simple 4-node traffic graph with travel times and capacities."""
    G = nx.Graph()

    edges = [
        ("A", "B", 5, 100),
        ("A", "D", 10, 150),
        ("A", "C", 6, 90),
        ("B", "C", 7, 80),
        ("B", "D", 3, 60),
        ("C", "D", 4, 120),
    ]

    for u, v, weight, cap in edges:
        G.add_edge(u, v, weight=weight, capacity=cap)

    return G

def generate_demands(G, num_demands=10, demand_size=25, seed=None):
    if seed is not None:
        random.seed(seed)

    nodes = list(G.nodes)
    demands = []
    for _ in range(num_demands):
        src, dst = random.sample(nodes, 2)
        demands.append((src, dst, demand_size))
    return demands


def enumerate_candidate_paths(G, demands, k=3):
    """
    For each demand (src, dst, demand_size), compute k candidate paths.
    Returns a list of candidate path lists.
    """
    all_candidates = []
    for (s, t, d) in demands:
        try:
            # Get up to k shortest simple paths
            paths = list(nx.shortest_simple_paths(G, s, t))[:k]
        except nx.NetworkXNoPath:
            paths = []
        all_candidates.append(paths)
    return all_candidates


def build_large_graph(grid_size=6, seed=42):
    """
    Build a larger grid network (grid_size x grid_size).
    Each edge has random capacity and unit length cost.
    """
    random.seed(seed)
    G = nx.grid_2d_graph(grid_size, grid_size)  # lattice network
    G = nx.convert_node_labels_to_integers(G)   # relabel 0..N-1
    
    for (u, v) in G.edges():
        G.edges[u, v]["capacity"] = random.randint(5, 15)
        G.edges[u, v]["weight"] = 1  # used for shortest path cost
    
    return G