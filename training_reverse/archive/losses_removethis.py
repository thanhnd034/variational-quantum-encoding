from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
import numpy as np

# These globals will be set externally by the training script (same as before)
qc = None
qc_app = None
qc_app_inv = None
target_sv = None
target_amp = None

# Projector |0><0|
projector_0 = DensityMatrix([[1, 0],
                             [0, 0]])


def loss_1q_mean_reverse(params):
    """
    Projector-based local loss used by Ben-Dov et al.
    Drop-in replacement for the previous RDM-based reverse loss.

    Computes:
        C(theta) = 1 - (1/n) * sum_j <0| rho_j(theta) |0>

    where rho_j(theta) is the reduced 1-qubit density matrix of qubit j
    after applying the inverse variational circuit to the target state.

    NOTE:
    - Same function name as original implementation
    - Same arguments and global dependencies
    - No other file needs modification
    """

    # Assign parameters to the inverse ansatz
    assignment = dict(zip(qc_app_inv.parameters, params))

    # Start from the precomputed exact statevector (set globally)
    sv = Statevector(target_amp)

    # Apply inverse variational circuit
    sv = sv.evolve(qc_app_inv.assign_parameters(assignment))

    N = qc_app_inv.num_qubits
    losses = []

    for q in range(N):
        # Partial trace out all other qubits
        trace_out = [i for i in range(N) if i != q]
        rho_q = partial_trace(sv, trace_out)  # single-qubit density matrix

        # Projector expectation: Tr[ rho_q |0><0| ]
        prob0 = np.real(np.trace(rho_q.data @ projector_0.data))

        # Local cost: 1 - p(0)
        C_q = 1.0 - prob0
        losses.append(C_q)

    # Mean over all qubits
    return np.mean(losses)
