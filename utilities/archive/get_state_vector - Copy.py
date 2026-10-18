from qiskit_aer import Aer
from qiskit import transpile

qc = None

def get_state_vector(qc):
    simulator = Aer.get_backend("statevector_simulator")
    transpiled_qc = transpile(qc,backend=simulator)
    job = simulator.run(transpiled_qc)
    result = job.result().get_statevector()
    target = result
    return target