from qiskit.quantum_info import Statevector
import numpy as np

training_params = None
qc_app = None
target = None

def loss_fidelity(params):
    # assign theta[i] = params[i]
    assignment = {training_params[i]: params[i] for i in range(len(training_params))}
    
    # get the output statevector
    sv = Statevector.from_instruction(qc_app.assign_parameters(assignment))
    
    # compute overlap <target | sv>
    overlap = np.vdot(target.data, sv.data)
    
    # fidelity loss = 1 - |<target|sv>|^2
    return 1 - np.abs(overlap)**2


def make_sparse_amplitude_loss(amplitude_indices):
    """
    Build a sparse amplitude MSE loss on a fixed subset of basis indices.

    amplitude_indices: 1D array-like of integer basis indices (0 .. 2^n - 1)
    """
    idx = np.array(amplitude_indices, dtype=int)
    idx = np.unique(idx)  # remove duplicates, however, this can reduce the number of amplitudes used

    def loss_sparse(params):
        # assign theta[i] = params[i]
        assignment = {training_params[i]: params[i] for i in range(len(training_params))}

        # output statevector for current parameters
        sv = Statevector.from_instruction(qc_app.assign_parameters(assignment))

        # difference only on selected amplitudes
        diff = sv.data[idx] - target.data[idx]

        # mean squared error on this subset (scaling doesn't matter for argmin)
        return float(np.mean(np.abs(diff) ** 2))

    return loss_sparse