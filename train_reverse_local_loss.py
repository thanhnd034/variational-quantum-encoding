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


print("=== Reverse 1 Qubit RDM Local Loss Training ===")

while True:
    folder_name = input("Enter the name of the dataset folder inside 'data_input': ").strip()
    data_number = input("Creating the circuit for data vector number: ").strip()
    number_iteration = input("Insert number of iteration for LBFGSB: ").strip()

    num_layers = default_number_of_layers

    # Call the helper
    dataset = load_dataset(folder_name)
    ip = dataset[int(data_number)]
    
    a,x = processData(ip)
    qc = build_sp_circuit(a,x)

    target = get_state_vector(qc)

    qc_app, training_params = build_app_circuit(int(np.log2(len(ip))), int(num_layers))
    print(f"\nNumber of parameters to train: {len(training_params)}")

    import training_global.losses as L

    L.training_params = training_params      # MUST come after build_app_circuit
    L.qc_app = qc_app
    L.target = target

    target_sv = Statevector.from_instruction(qc)  # <-- compute qc output one time
    target_amp = target_sv.data                  # <-- amplitude vector
    qc_app_inv = qc_app.inverse()                # <-- cached inverse circuit

    import training_reverse.losses as M
    M.qc_app = qc_app
    M.qc = qc
    M.target_sv = target_sv
    M.target_amp = target_amp
    M.qc_app_inv = qc_app_inv

    from training_reverse.losses import loss_1q_mean_reverse
    print()

    initial_params = generate_initial_params(len(training_params), seed=make_seed("initial data generation"))

    result_app = optimize_with_lbfgsb(loss_1q_mean_reverse, initial_params, maxiter=int(number_iteration))

    print("Optimal parameters:", result_app.x)
    print("Final loss:", result_app.fun)
    
    output_data(folder_name, data_number, result_app)

    cont = input("\nTrain another data vector? (y/n): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("\nExiting trainer.")
        break

    print()