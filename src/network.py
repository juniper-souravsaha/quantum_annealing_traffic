import networkx as nx

def create_grid_network(rows=4, cols=4, capacity=10, length=1):
    G = nx.grid_2d_graph(rows, cols)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]['capacity'] = capacity
        G[u][v]['length'] = length
    return G
