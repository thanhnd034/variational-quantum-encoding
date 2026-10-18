from qiskit.quantum_info import partial_trace, state_fidelity
from qiskit.quantum_info import Statevector

# These are set externally by your training script
training_params = None
qc_app = None
target = None

# Cache for target reduced density matrices:
# key = tuple(subset), value = target_rho
_TARGET_RDM_CACHE = {}


def _get_target_rdm(subset):
    """
    Return cached target RDM for this subset.
    If not cached, compute it once and store it.
    """
    subset = tuple(subset)

    if subset in _TARGET_RDM_CACHE:
        return _TARGET_RDM_CACHE[subset]

    # compute once
    num_qubits = qc_app.num_qubits
    trace_out = [q for q in range(num_qubits) if q not in subset]
    target_rho = partial_trace(target, trace_out)

    _TARGET_RDM_CACHE[subset] = target_rho
    return target_rho


def local_rdm_loss(params, subset):
    """
    Compute 1 - fidelity( RDM(params), RDM(target) )
    subset: list of qubit indices to keep
    """

    # assign parameters to the approximate circuit
    assignment = {training_params[i]: params[i] for i in range(len(training_params))}
    sv = Statevector.from_instruction(qc_app.assign_parameters(assignment))

    # compute approximate circuit RDM
    num_qubits = qc_app.num_qubits
    trace_out = [q for q in range(num_qubits) if q not in subset]
    rho = partial_trace(sv, trace_out)

    # fetch precomputed target RDM (never recomputed again)
    target_rho = _get_target_rdm(subset)

    # return local RDM fidelity loss
    return 1 - state_fidelity(rho, target_rho)
