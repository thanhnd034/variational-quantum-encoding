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
import training_global.losses as L
from utilities.output_data import output_data

print("=== Global Fidelity Training ===")

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

    print()
    initial_params = generate_initial_params(len(training_params), seed=make_seed("initial data generation"))

    result_app = optimize_with_lbfgsb(loss_fidelity, initial_params, maxiter=int(number_iteration))

    print("Optimal parameters:", result_app.x)
    print("Final loss:", result_app.fun)
    
    output_data(folder_name, data_number, result_app)

    cont = input("\nTrain another data vector? (y/n): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("\nExiting trainer.")
        break

    print()