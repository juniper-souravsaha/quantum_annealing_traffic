import math
import random
from src import traffic_sim, metrics
import networkx as nx

def simulated_annealing(G, demand, temp_start, temp_end, episodes):
    current_paths = traffic_sim.assign_paths_randomly(G, demand)
    current_loads = traffic_sim.compute_edge_loads(G, current_paths)
    current_cost = metrics.cost_function(G, current_loads)
    best_paths, best_cost = current_paths[:], current_cost

    temp = temp_start
    # Slower cooling
    cooling_rate = (temp_end / temp_start) ** (1.0 / (episodes * 1.5))

    for ep in range(episodes):
        # Pick a vehicle to re-route
        new_paths = current_paths[:]
        idx = random.randint(0, len(demand) - 1)
        start, end = demand[idx]

        try:
            # Get top k shortest simple paths and pick one at random
            k = 5
            all_paths = list(nx.shortest_simple_paths(G, start, end, weight='length'))
            if len(all_paths) > 1:
                alt_path = random.choice(all_paths[:k])
                new_paths[idx] = alt_path
        except nx.NetworkXNoPath:
            pass

        new_loads = traffic_sim.compute_edge_loads(G, new_paths)
        new_cost = metrics.cost_function(G, new_loads)

        # Accept with probability
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temp):
            current_paths, current_cost = new_paths, new_cost
            if new_cost < best_cost:
                best_paths, best_cost = new_paths[:], new_cost

        temp *= cooling_rate

        if ep % max(1, episodes // 10) == 0:
            print(f"Episode {ep}: Current cost = {current_cost:.2f}, Best cost = {best_cost:.2f}")

    return best_paths, best_cost