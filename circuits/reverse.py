from qiskit.circuit import QuantumCircuit, ParameterVector

def build_reverse_circuit_manual(num_qubits, num_layers, training_params):
    qc_rev = QuantumCircuit(num_qubits)

    for layer in reversed(range(num_layers)):
        base = num_qubits * 2 * layer
        cry_base = base + num_qubits

        # inverse CRY
        for q in reversed(range(num_qubits)):
            control = q
            target_qubit = 0 if q == num_qubits - 1 else q + 1
            qc_rev.cry(-training_params[cry_base + q], control, target_qubit)

        # inverse RY
        for q in reversed(range(num_qubits)):
            qc_rev.ry(-training_params[base + q], q)

        qc_rev.barrier()

    return qc_rev
