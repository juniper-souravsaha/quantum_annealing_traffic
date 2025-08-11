import networkx as nx
import matplotlib.pyplot as plt

def create_test_network():
    G = nx.Graph()
    edges = [
        ("A", "B", 1),
        ("B", "C", 1),
        ("C", "D", 1),
        ("A", "D", 3),  # longer direct route
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w, load=0)
    return G

def assign_traffic(G):
    # traffic demands (source, target, demand)
    demands = [("A", "D", 10), ("B", "C", 5)]
    for src, dst, demand in demands:
        path = nx.shortest_path(G, src, dst, weight="weight")
        print(f"Shortest path {src} → {dst}: {path}")
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            G[u][v]["load"] += demand

def plot_network(G):
    pos = nx.spring_layout(G, seed=42)
    loads = [G[u][v]["load"] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{G[u][v]['load']} load" for u, v in G.edges()})
    edges = nx.draw_networkx_edges(G, pos, width=[load for load in loads], edge_color=loads, edge_cmap=plt.cm.plasma)
    plt.colorbar(edges)
    plt.title("Test Network with Loads")
    plt.show()

if __name__ == "__main__":
    G = create_test_network()
    assign_traffic(G)
    plot_network(G)