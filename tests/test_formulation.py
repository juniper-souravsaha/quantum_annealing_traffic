import networkx as nx

from src.formulation import (compute_edge_loads_from_state,
                             k_shortest_candidates, objective_cost,
                             random_initial_state)


def tiny_graph():
    G = nx.Graph()
    G.add_edge("A","B", time=1, capacity=2)
    G.add_edge("B","C", time=1, capacity=2)
    G.add_edge("A","C", time=3, capacity=2)
    return G

def test_loads_and_cost():
    G = tiny_graph()
    demand = [("A","C"), ("A","C")]
    cands = k_shortest_candidates(G, demand, k=2)  # paths: A-B-C and A-C
    # state 0: both use A-B-C
    state = [0, 0]
    loads = compute_edge_loads_from_state(G, cands, state)
    assert loads[tuple(sorted(("A","B")))] == 2
    assert loads[tuple(sorted(("B","C")))] == 2
    assert loads[tuple(sorted(("A","C")))] == 0

    # cost with low penalty should prefer A-B-C (time=2 vs 3)
    c_low = objective_cost(G, cands, state, congestion_penalty_coef=0.0)
    assert c_low == 4  # two trips, each time=2

    # with high penalty and very low capacity, it should grow
    G["A"]["B"]["capacity"] = 1
    G["B"]["C"]["capacity"] = 1
    c_high = objective_cost(G, cands, state, congestion_penalty_coef=10.0)
    assert c_high > c_low