import dimod


# D-Wave solver
def solve_dwave(bqm, num_reads=100):
    from dwave.system import DWaveSampler, EmbeddingComposite
    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    return sampleset.first.sample, sampleset.first.energy


# QAOA solver
def solve_qaoa(bqm, reps=3, maxiter=200):
    import docplex.mp.model as cpx
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.primitives import Estimator
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization.translators import from_docplex_mp

    # Convert BQM → Qiskit QuadraticProgram
    mdl = cpx.Model()
    var_map = {}
    for v in bqm.variables:
        var_map[v] = mdl.binary_var(name=v)
    # Objective: from linear+quadratic dicts
    obj = mdl.sum(bqm.linear[v]*var_map[v] for v in bqm.linear)
    for (u,v), w in bqm.quadratic.items():
        obj += w*var_map[u]*var_map[v]
    mdl.minimize(obj)

    qp = from_docplex_mp(mdl)

    qaoa = QAOA(Estimator(), optimizer=COBYLA(maxiter=maxiter), reps=reps)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)
    return result.x, result.fval