from qiskit.quantum_info import Statevector


def get_state_vector(qc):
    """
    Return the statevector of the circuit qc.
    Avoid transpiling because the Qiskit Rust transpiler can panic
    for some circuits on this HPC installation.
    """
    return Statevector.from_instruction(qc)
