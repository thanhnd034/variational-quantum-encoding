# train_reverse_local_loss_multi.py
#
# Same training as train_reverse_local_loss.py,
# but now loops automatically from vector i_start to i_end.

from utilities.load_dataset import load_dataset
from circuits.helpers import processData
from circuits.approx import build_app_circuit
import numpy as np

from training_global.losses import loss_fidelity
from maths.optimize_with_lbfgsb import optimize_with_lbfgsb
from utilities.make_seed import make_seed
from utilities.initial_params import generate_initial_params
from configs.settings import default_number_of_layers

from circuits.original import build_sp_circuit
from utilities.get_state_vector import get_state_vector
from utilities.output_data import output_data

from qiskit.quantum_info import Statevector


print("=== Reverse 1 Qubit RDM Local Loss Training (MULTI-VECTOR) ===")

# ---------------------------
# INPUTS
# ---------------------------
folder_name = input("Enter dataset folder inside 'data_input': ").strip()
i_start = int(input("Train from data vector index: ").strip())
i_end   = int(input("Train to   data vector index: ").strip())
number_iteration = int(input("Insert number of iteration for LBFGSB: ").strip())

num_layers = default_number_of_layers

# Load entire dataset once
dataset = load_dataset(folder_name)

# ---------------------------
# MULTI-VECTOR LOOP
# ---------------------------
for data_number in range(i_start, i_end + 1):

    print(f"\n=== Training data vector #{data_number} ===")

    ip = dataset[data_number]

    # build original circuit
    a, x = processData(ip)
    qc = build_sp_circuit(a, x)

    target = get_state_vector(qc)

    # build approximation circuit
    n_qubits = int(np.log2(len(ip)))
    qc_app, training_params = build_app_circuit(n_qubits, int(num_layers))
    print(f"Number of parameters to train: {len(training_params)}")

    # set global loss module for fidelity
    import training_global.losses as L
    L.training_params = training_params
    L.qc_app = qc_app
    L.target = target

    # precompute static values for reverse loss
    target_sv = Statevector.from_instruction(qc)
    target_amp = target_sv.data
    qc_app_inv = qc_app.inverse()

    # set reverse-loss globals
    import training_reverse.losses as M
    M.qc_app = qc_app
    M.qc = qc
    M.target_sv = target_sv
    M.target_amp = target_amp
    M.qc_app_inv = qc_app_inv

    from training_reverse.losses import loss_1q_mean_reverse

    # initial parameters
    initial_params = generate_initial_params(
        len(training_params),
        seed=make_seed(f"initial_params_vec{data_number}")
    )

    # run training
    result_app = optimize_with_lbfgsb(
        loss_1q_mean_reverse,
        initial_params,
        maxiter=number_iteration
    )

    print(f"  Final loss: {result_app.fun}")
    print(f"  Saving results...")

    output_data(folder_name, data_number, result_app)

print("\n=== Finished training all vectors. ===")
