from dimod import SimulatedAnnealingSampler
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
import docplex.mp.model as cpx
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_aer import AerSimulator

# D-Wave solver
def solve_dwave(bqm, num_reads=100):
    """
    Solve a BQM using D-Wave sampler if available,
    otherwise fall back to a local simulated annealer.
    """
    try:
        # Try using the actual quantum hardware
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample(bqm, num_reads=num_reads)
        sol = response.first.sample
        energy = response.first.energy
        print("✅ Solved using D-Wave QPU.")
    except Exception as e:
        print(f"⚠️ Falling back to Simulated Annealing (local). Reason: {e}")
        sampler = SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=num_reads)
        sol = response.first.sample
        energy = response.first.energy

    return sol, energy


# ---------- Helper: dimod BQM -> Qiskit QuadraticProgram ----------
def _bqm_to_quadratic_program(bqm: dimod.BinaryQuadraticModel):
    qp = QuadraticProgram()
    name_map = {}
    for v in bqm.variables:
        name_map[v] = str(v)
        qp.binary_var(name=name_map[v])

    linear = {name_map[v]: float(b) for v, b in bqm.linear.items()}
    quad = {(name_map[u], name_map[v]): float(w) for (u, v), w in bqm.quadratic.items()}

    const = float(bqm.offset) if bqm.offset is not None else 0.0
    qp.minimize(constant=const, linear=linear, quadratic=quad)
    return qp

# ---------- QAOA (with graceful fallback & size cap) ----------
def solve_qaoa(bqm, reps=1):
    Q, offset = bqm.to_qubo()

    qp = QuadraticProgram()
    for v in bqm.variables:
        qp.binary_var(name=str(v))

    print(f"QUBO size: {len(Q)} terms")
    linear = {str(i): Q.get((i, i), 0.0) for i in bqm.variables}
    quadratic = {(str(i), str(j)): coeff for (i, j), coeff in Q.items() if i != j}
    qp.minimize(linear=linear, quadratic=quadratic)

    print(f"Converted to QuadraticProgram with {qp.get_num_vars()} variables and {qp.get_num_linear_constraints()} constraints.")
    # --- Fixed backend setup ---
    backend = AerSimulator()
    # backend.set_options(shots=128)
    sampler = Sampler(options={"shots": 1})

    print("Setting up QAOA...")
    qaoa = QAOA(
        sampler=sampler,
        reps=reps,
        optimizer=COBYLA(maxiter=5)
    )
    optimizer = MinimumEigenOptimizer(qaoa)

    result = optimizer.solve(qp)
    return result.x, result.fval