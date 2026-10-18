from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RYGate
from qiskit.circuit import ParameterVector, Parameter
import numpy as np

def build_app_circuit(num_qubits, num_layers):

    qc_app = QuantumCircuit(num_qubits)
    n_params = num_layers * num_qubits * 2
    training_params = ParameterVector("θ", n_params)

    for layer in range(num_layers):
        base = num_qubits * 2 * layer   # starting index for this layer

        # --- RY block ---
        for q in range(num_qubits):
            qc_app.ry(training_params[base + q], q)

        # --- CRY entangler block ---
        cry_base = base + num_qubits  # start of CRY params
        for q in range(num_qubits):
            control = q
            target_qubit = 0 if q == num_qubits - 1 else q + 1
            qc_app.cry(training_params[cry_base + q], control, target_qubit)

        qc_app.barrier()

    return qc_app, training_params