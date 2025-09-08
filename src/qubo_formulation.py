import dimod

def build_qubo(G, demands, candidate_lists, alpha=1.0, beta=1.0):
    """
    Build normalized QUBO for traffic assignment.
    - G: networkx graph with 'capacity' on edges
    - demands: list of (src, dst, demand_val)
    - candidate_lists: list of candidate paths per demand
    - alpha, beta: penalty weights (scaled automatically)
    """
    bqm = dimod.BinaryQuadraticModel('BINARY')

    # Scale penalties relative to max demand/capacity
    max_demand = max(d[2] for d in demands)
    max_cap = max(G[e[0]][e[1]].get("capacity", 1) for e in G.edges)
    alpha_scaled = alpha * max_demand
    beta_scaled = beta / max_cap

    print(f"[QUBO] Scaling factors → alpha={alpha_scaled:.3f}, beta={beta_scaled:.3f}")
    print(f"[QUBO] Max demand={max_demand}, Max capacity={max_cap}")

    # One-path-per-demand constraint
    for d, (src, dst, dem) in enumerate(demands):
        paths = candidate_lists[d]
        vars_d = [f"x_{d}_{i}" for i in range(len(paths))]

        # (1 - sum(x))^2 penalty
        bqm.offset += alpha_scaled
        for v in vars_d:
            bqm.add_variable(v, -2 * alpha_scaled)
        for i in range(len(vars_d)):
            for j in range(i+1, len(vars_d)):
                bqm.add_interaction(vars_d[i], vars_d[j], 2 * alpha_scaled)

    # Capacity violation penalties
    for e in G.edges:
        cap = G[e[0]][e[1]].get("capacity", 1)
        load_expr = []
        for d, (src, dst, dem) in enumerate(demands):
            for i, path in enumerate(candidate_lists[d]):
                if e in path:
                    load_expr.append((dem, f"x_{d}_{i}"))

        if load_expr:
            for coef, v in load_expr:
                bqm.add_variable(v, beta_scaled * coef**2)
                for coef2, v2 in load_expr:
                    if v != v2:
                        bqm.add_interaction(v, v2, 2 * beta_scaled * coef * coef2)

            bqm.offset += beta_scaled * cap**2
            for coef, v in load_expr:
                bqm.add_variable(v, -2 * beta_scaled * cap * coef)

    return bqm