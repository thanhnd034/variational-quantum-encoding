from utilities.load_dataset import load_dataset
from circuits.helpers import processData
from circuits.approx import build_app_circuit
import numpy as np
from training_global.losses import loss_fidelity
from utilities.make_seed import make_seed
from utilities.initial_params import generate_initial_params
from configs.settings import default_number_of_layers
from circuits.original import build_sp_circuit
from utilities.get_state_vector import get_state_vector
import training_global.losses as L
from utilities.output_data import output_data
from utilities.make_seed import make_seed
from training_local.train_rdm_mean import make_kq_rdm_loss, train_rdm_lbfgsb


print("=== Local RDM Mean Training ===")

while True:
    folder_name = input("Enter the name of the dataset folder inside 'data_input': ").strip()
    data_number = input("Creating the circuit for data vector number: ").strip()
    number_iteration = input("Insert number of iteration for LBFGSB: ").strip()
    num_qubit_rdm = input("Insert number of qubits used for mean RDM loss: ").strip()

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

    import training_local.train_rdm_mean as T

    T.qc_app = qc_app 

    loss_1q_mean = make_kq_rdm_loss(1)
    loss_2q_mean = make_kq_rdm_loss(2)
    loss_3q_mean = make_kq_rdm_loss(3)

    print()
    initial_params = generate_initial_params(len(training_params), seed=make_seed("initial data generation"))

    if num_qubit_rdm == 1:
        result_app =   train_rdm_lbfgsb(initial_params,
                                                loss_fn=loss_1q_mean,
                                                maxiter=int(number_iteration),     
                                                label="1q mean RDM")

    elif num_qubit_rdm == 2:
        result_app =   train_rdm_lbfgsb(initial_params,
                                                loss_fn=loss_2q_mean,
                                                maxiter=int(number_iteration),     
                                                label="2q mean RDM")

    elif num_qubit_rdm == 3:
        result_app =   train_rdm_lbfgsb(initial_params,
                                                loss_fn=loss_3q_mean,
                                                maxiter=int(number_iteration),     
                                                label="1q mean RDM")
        
    else:
        loss_nq_mean = make_kq_rdm_loss(int(num_qubit_rdm))
        result_app =   train_rdm_lbfgsb(initial_params,
                                                loss_fn=loss_nq_mean,
                                                maxiter=int(number_iteration),     
                                                label=f"{num_qubit_rdm}q mean RDM")

    

    # find index of entry with smallest global_loss
    global_fids = result_app.history["global_fidelity"]
    best_idx = int(np.argmin(global_fids))

    best_cycle  = best_idx + 1
    best_loss   = result_app.history["global_fidelity"][best_idx]
    best_params = result_app.history["params"][best_idx]

    print("Best cycle:", best_cycle)
    print("Best global_loss:", best_loss)
    
    output_data(folder_name, data_number, result_app)

    cont = input("\nTrain another data vector? (y/n): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("\nExiting trainer.")
        break

    print()