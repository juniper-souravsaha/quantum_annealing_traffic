import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_network(G, best_paths):
    pos = nx.spring_layout(G)
    
    # Compute edge traffic load
    edge_load = {edge: 0 for edge in G.edges()}
    for path in best_paths:
        for u, v in zip(path[:-1], path[1:]):
            if (u, v) in edge_load:
                edge_load[(u, v)] += 1
            elif (v, u) in edge_load:
                edge_load[(v, u)] += 1
    
    # Prepare color mapping
    loads = np.array(list(edge_load.values()))
    norm = Normalize(vmin=loads.min(), vmax=loads.max())
    cmap = plt.cm.viridis
    
    # Create plot
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, ax=ax,
            edge_color=[cmap(norm(load)) for load in loads],
            width=2, node_size=500)
    
    # Create ScalarMappable for colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Traffic Load')
    
    plt.show()