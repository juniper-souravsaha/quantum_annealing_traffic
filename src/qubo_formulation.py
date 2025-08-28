import dimod


def build_qubo(G, demands, candidate_lists, alpha=5.0, beta=10.0):
    """
    Build QUBO for traffic assignment.
    - G: networkx graph with 'capacity' on edges
    - demands: list of (src, dst, demand_val)
    - candidate_lists: dict[d] -> list of candidate paths (each path = list of edges)
    - alpha, beta: penalty weights
    """
    bqm = dimod.BinaryQuadraticModel('BINARY')

    # One-path-per-demand constraint
    for d, (src, dst, dem) in enumerate(demands):
        paths = candidate_lists[d]
        vars_d = [f"x_{d}_{i}" for i in range(len(paths))]

        # (1 - sum(x))^2
        bqm.offset += alpha
        for v in vars_d:
            bqm.add_variable(v, -2 * alpha)
        for i in range(len(vars_d)):
            for j in range(i+1, len(vars_d)):
                bqm.add_interaction(vars_d[i], vars_d[j], 2 * alpha)

    # Capacity violation penalties
    for e in G.edges:
        cap = G[e[0]][e[1]].get("capacity", 1)
        load_expr = []
        for d, (src, dst, dem) in enumerate(demands):
            for i, path in enumerate(candidate_lists[d]):
                if e in path:
                    load_expr.append((dem, f"x_{d}_{i}"))
        # quadratic penalty (sum(demand*x) - cap)^2
        for coef, v in load_expr:
            bqm.add_variable(v, beta * coef**2)
            for coef2, v2 in load_expr:
                if v != v2:
                    bqm.add_interaction(v, v2, 2 * beta * coef * coef2)
        bqm.offset += beta * cap**2
        for coef, v in load_expr:
            bqm.add_variable(v, -2 * beta * cap * coef)

    return bqm