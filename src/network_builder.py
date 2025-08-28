import networkx as nx

from .network_config import ROAD_NETWORK


def build_network():
    """Builds and returns a NetworkX graph from ROAD_NETWORK."""
    G = nx.Graph()
    for src, dst, attrs in ROAD_NETWORK:
        G.add_edge(src, dst, **attrs)
    return G