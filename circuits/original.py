import numpy as np
from circuits.helpers import calculateRy, repositionVector, int_to_reversed_bitstring, make_list
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RYGate

def build_sp_circuit(a, x):
    """
    Build the SP (staircase) amplitude-encoding circuit using your original code.

    Args:
        a : amplitude vector
        x : input vector for calculateRy
        calculateRy : function x -> raw Ry angles
        repositionVector : function to reorder the angle vector
        int_to_reversed_bitstring : function(j, i) -> control state string
        make_list : function(i) -> list of qubit indices for qc.append

    Returns:
        qc : QuantumCircuit
    """

    c = np.array(calculateRy(x))
    Ry_angles = repositionVector(c)

    qc = QuantumCircuit(np.log2(len(a)))
    qc.ry(Ry_angles[0],0)
    for i in range(1, int(np.log2(len(a)))):
        for j in range(2**i):
            if Ry_angles[(2**i)-1+j] == 0 or np.isnan(Ry_angles[(2**i)-1+j]):
                continue
            ry = RYGate(Ry_angles[(2**i)-1+j]).control(i, ctrl_state=int_to_reversed_bitstring(j,i))
            qc.append(ry,make_list(i))

    return qc