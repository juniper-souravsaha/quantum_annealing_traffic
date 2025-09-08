import logging
from dimod import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
# import docplex.mp.model as cpx
from qiskit_algorithms import QAOA
from qiskit.primitives import Estimator
from qiskit.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.optimizers import COBYLA,SPSA
from qiskit_aer import AerSimulator
from qiskit_optimization.algorithms import MinimumEigenOptimizer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # adjust as desired

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


def solve_sa(bqm, num_reads=100):
    """
    Solve QUBO using Classical Simulated Annealing (SA).
    """
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    best = sampleset.first
    return best.sample, best.energy


def bqm_to_qp(bqm):
    """Convert dimod.BinaryQuadraticModel -> Qiskit QuadraticProgram (robust)."""
    qp = QuadraticProgram("traffic_qubo")

    # Add binary variables
    for v in bqm.variables:
        qp.binary_var(name=str(v))

    # Linear coefficients
    linear = {str(v): float(bias) for v, bias in bqm.linear.items()}

    # Quadratic coefficients
    quadratic = {(str(u), str(v)): float(bias) for (u, v), bias in bqm.quadratic.items()}

    # Offset / constant
    constant = float(bqm.offset) if getattr(bqm, "offset", 0) is not None else 0.0

    # Set objective (minimization)
    qp.minimize(linear=linear, quadratic=quadratic, constant=constant)
    return qp


def _qp_constraint_counts(qp: QuadraticProgram):
    """Return (num_vars, num_linear_constraints, num_quadratic_constraints) robustly."""
    # num_vars
    if hasattr(qp, "get_num_vars"):
        num_vars = qp.get_num_vars()
    else:
        num_vars = len(qp.variables)

    # linear constraints
    if hasattr(qp, "get_num_linear_constraints"):
        num_lin = qp.get_num_linear_constraints()
    else:
        # fallback to attribute (older/newer variations)
        num_lin = len(getattr(qp, "linear_constraints", []))

    # quadratic constraints
    if hasattr(qp, "get_num_quadratic_constraints"):
        num_quad = qp.get_num_quadratic_constraints()
    else:
        num_quad = len(getattr(qp, "quadratic_constraints", []))

    return int(num_vars), int(num_lin), int(num_quad)


def solve_qaoa(bqm, reps=1, maxiter=50, optimizer_name="SPSA"):
    """
    Solve dimod BQM using QAOA via QuadraticProgram conversion.
    - Robust prints for qp sizes/constraints compatible with multiple qiskit versions.
    - Returns (solution_dict, objective_value)
    """
    # ---- convert BQM -> QuadraticProgram ----
    qp = bqm_to_qp(bqm)

    # robust constraint/var counts
    num_vars, num_lin, num_quad = _qp_constraint_counts(qp)
    log.info(f"[QAOA] QuadraticProgram: vars={num_vars}, linear_constraints={num_lin}, quadratic_constraints={num_quad}")

    # variable name list for mapping
    var_names = [v.name for v in qp.variables]
    log.info(f"[QAOA] Variable names ({len(var_names)}): {var_names}")

    # ---- pick optimizer object ----
    if optimizer_name.upper().startswith("SPSA"):
        optimizer_obj = SPSA(maxiter=maxiter)
    else:
        optimizer_obj = COBYLA(maxiter=maxiter)

    # ---- QAOA setup ----
    try:
        qaoa = QAOA(
            sampler=Sampler(),
            reps=reps,
            optimizer=optimizer_obj,
            initial_point=np.random.rand(2 * reps)  # random init helps avoid trivial basin
        )
        meo = MinimumEigenOptimizer(qaoa)

        log.info(f"[QAOA] Running QAOA (reps={reps}, maxiter={maxiter}, optimizer={optimizer_name}) ...")
        result = meo.solve(qp)

    except Exception as e:
        log.exception("[QAOA] QAOA/MinimumEigenOptimizer failed")
        raise

    # ---- parse result safely ----
    x = getattr(result, "x", None)
    fval = getattr(result, "fval", getattr(result, "fval", None))

    if x is None:
        # older result objects may carry .x as list-like under different attr
        try:
            x = list(result.samples[0].x)  # fallbacks (not guaranteed)
        except Exception:
            log.error("[QAOA] Could not extract solution vector from result object.")
            raise RuntimeError("Could not parse QAOA result.x")

    # Map names -> integer values (0/1)
    solution = {}
    for i, name in enumerate(var_names):
        # some result.x might be floats (0.0 or 1.0), convert safely
        val = int(round(float(x[i])))
        solution[name] = val

    log.info(f"[QAOA] Parsed solution (first 20 shown): {dict(list(solution.items())[:20])}")
    log.info(f"[QAOA] Objective value: {fval}")

    return solution, float(fval)
