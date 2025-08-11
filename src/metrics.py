def cost_function(G, loads):
    cost = 0
    for (u, v), load in loads.items():
        length = G[u][v]['length']
        capacity = G[u][v]['capacity']
        congestion_penalty = max(0, load - capacity) * 5
        cost += length * load + congestion_penalty
    return cost