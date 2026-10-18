from utilities.load_dataset import load_dataset
from circuits.helpers import processData
from circuits.approx import build_app_circuit
import numpy as np
from training_global.losses import loss_fidelity
from maths.optimize_with_lbfgsb import optimize_with_lbfgsb
from utilities.make_seed import make_seed
from utilities.initial_params import generate_initial_params
from configs.settings import default_number_of_layers, default_lr_schedule, default_delta_schedule, default_seed_schedule
from circuits.original import build_sp_circuit
from utilities.get_state_vector import get_state_vector
import training_global.losses as L
from utilities.output_data2 import output_data2
from configs.global_subset_cycle import get_subset_size_for_cycle
from utilities.make_seed import make_seed


print("=== Global Fidelity Training with Schedule ===")

while True:
    folder_name = input("Enter the name of the dataset folder inside 'data_input': ").strip()
    data_number = input("Creating the circuit for data vector number: ").strip()
    number_iteration = input("Insert number of iteration for Adam: ").strip()

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

    import training_global.train_sparse_fidelity_with_schedule as T
    T.qc_app = qc_app
    from training_global.train_sparse_fidelity_with_schedule import random_subset_fidelity_training_adam

    print()
    initial_params = generate_initial_params(len(training_params), seed=make_seed("initial data generation"))

    result_app, history = random_subset_fidelity_training_adam(initial_params, num_cycles=int(number_iteration), block_size=30, batch_size_amplitudes=get_subset_size_for_cycle, lr=default_lr_schedule, delta=default_delta_schedule, seed=make_seed(default_seed_schedule))

    # find index of entry with smallest global_loss
    best_idx = np.argmin([entry["global_loss"] for entry in history])

    best_cycle = best_idx + 1  # +1 because cycles were printed starting from 1
    best_loss  = history[best_idx]["global_loss"]
    best_params = history[best_idx]["params"]

    print("Best cycle:", best_cycle)
    print("Best global_loss:", best_loss)
    
    output_data2(folder_name, data_number, best_params, history)

    cont = input("\nTrain another data vector? (y/n): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("\nExiting trainer.")
        break

    print()