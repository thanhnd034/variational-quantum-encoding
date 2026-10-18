from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity
import numpy as np

qc = None
qc_app = None


target_1q_reverse = DensityMatrix([[1, 0],
                                   [0, 0]])

# ----------------------------------------------------
# Precompute target state from qc   (RUN ONCE)
# ----------------------------------------------------
target_sv = None  # <-- compute qc output one time
target_amp = None                  # <-- amplitude vector
qc_app_inv = None                # <-- cached inverse circuit
# ----------------------------------------------------

def loss_1q_mean_reverse(params):
    # Only assign parameters that qc_app_inv actually has
    assignment = dict(zip(qc_app_inv.parameters, params))

    # Start from precomputed target amplitude
    sv = Statevector(target_amp)

    # Apply only inverse of variational circuit
    sv = sv.evolve(qc_app_inv.assign_parameters(assignment))

    # 1q RDM losses
    N = qc_app_inv.num_qubits
    losses = []

    for q in range(N):
        trace_out = [i for i in range(N) if i != q]
        rho_q = partial_trace(sv, trace_out)
        losses.append(1 - state_fidelity(rho_q, target_1q_reverse))

    return np.mean(losses)
