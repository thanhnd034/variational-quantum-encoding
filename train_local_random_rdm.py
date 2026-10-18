from utilities.load_dataset import load_dataset
from circuits.helpers import processData
from circuits.approx import build_app_circuit
import numpy as np
from training_global.losses import loss_fidelity
from utilities.make_seed import make_seed
from utilities.initial_params import generate_initial_params
from configs.settings import default_number_of_layers, default_delta_adam_random_rdm, default_lr_adam_random_rdm
from circuits.original import build_sp_circuit
from utilities.get_state_vector import get_state_vector
import training_global.losses as L
from utilities.output_data3 import output_data3
from utilities.make_seed import make_seed
from training_local.losses import local_rdm_loss


print("=== Local RDM Random Training ===")

while True:
    folder_name = input("Enter the name of the dataset folder inside 'data_input': ").strip()
    data_number = input("Creating the circuit for data vector number: ").strip()
    number_iteration = input("Insert number of iteration for Adam: ").strip()
    num_qubit_rdm = input("Insert number of random qubits used for RDM: ").strip()

    num_layers = default_number_of_layers

    # Call the helper
    dataset = load_dataset(folder_name)
    ip = dataset[int(data_number)]
    
    a,x = processData(ip)
    qc = build_sp_circuit(a,x)

    target = get_state_vector(qc)

    qc_app, training_params = build_app_circuit(int(np.log2(len(ip))), int(num_layers))
    print(f"\nNumber of parameters to train: {len(training_params)}")

    import training_local.losses as L

    L.training_params = training_params      # MUST come after build_app_circuit
    L.qc_app = qc_app
    L.target = target

    import training_global.losses as M

    M.training_params = training_params      # MUST come after build_app_circuit
    M.qc_app = qc_app
    M.target = target

    import training_local.train_rdm_random_adam as T
    T.qc_app = qc_app
    from training_local.train_rdm_random_adam import RDMTrainerAdam

    trainer = RDMTrainerAdam(
    qc_app=qc_app,
    local_rdm_loss=local_rdm_loss,
    loss_fidelity=loss_fidelity,
    default_lr=0.01,
    default_delta=1e-5)


    print()
    initial_params = generate_initial_params(len(training_params), seed=make_seed("initial data generation"))

    if num_qubit_rdm == 1:
        result_app, history =   trainer.train_1q(initial_params,
                                                num_cycles=int(number_iteration),     
                                                block_size=len(training_params),
                                                lr=default_lr_adam_random_rdm,
                                                delta=default_delta_adam_random_rdm,
                                                seed=make_seed("train_one_q"))

    elif num_qubit_rdm == 2:
        result_app, history =   trainer.train_2q(initial_params,
                                                num_cycles=int(number_iteration),     
                                                block_size=len(training_params),
                                                lr=default_lr_adam_random_rdm,
                                                delta=default_delta_adam_random_rdm,
                                                seed=make_seed("train_two_q"))

    elif num_qubit_rdm == 3:
        result_app, history =   trainer.train_3q(initial_params,
                                                num_cycles=int(number_iteration),     
                                                block_size=len(training_params),
                                                lr=default_lr_adam_random_rdm,
                                                delta=default_delta_adam_random_rdm,
                                                seed=make_seed("train_three_q"))
        
    else:
        result_app, history =   trainer.train_nq(int(num_qubit_rdm),
                                                initial_params,
                                                num_cycles=int(number_iteration),     
                                                block_size=len(training_params),
                                                lr=default_lr_adam_random_rdm,
                                                delta=default_delta_adam_random_rdm,
                                                seed=make_seed("train_large_number_q"))

    

    # find index of entry with smallest global_loss
    best_idx = np.argmin([entry["global_loss"] for entry in history])

    best_cycle = best_idx + 1  # +1 because cycles were printed starting from 1
    best_loss  = history[best_idx]["global_loss"]
    best_params = history[best_idx]["params"]

    print("Best cycle:", best_cycle)
    print("Best global_loss:", best_loss)
    
    output_data3(folder_name, data_number, best_params, history)

    cont = input("\nTrain another data vector? (y/n): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("\nExiting trainer.")
        break

    print()