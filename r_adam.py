"""
Train reverse circuit using local 1q RDM loss + ADAM optimizer
"""

from utilities.load_dataset import load_dataset
from circuits.helpers import processData
from circuits.approx import build_app_circuit
import numpy as np
from utilities.make_seed import make_seed
from utilities.initial_params import generate_initial_params
from configs.settings import default_number_of_layers
from circuits.original import build_sp_circuit
from utilities.get_state_vector import get_state_vector
from utilities.output_data import output_data
from qiskit.quantum_info import Statevector

# Adam + gradient
from maths.numerical_grad import numerical_grad
from maths.Adam import Adam
from training_global.losses import loss_fidelity
from training_reverse.losses import loss_1q_mean_reverse


print("=== Reverse 1 Qubit RDM Local Loss Training (ADAM) ===")

while True:
    folder_name = input("Enter the dataset folder: ").strip()
    data_number = input("Creating the circuit for data vector number: ").strip()
    num_steps = int(input("Number of ADAM steps: ").strip())

    num_layers = default_number_of_layers

    # load dataset
    dataset = load_dataset(folder_name)
    ip = dataset[int(data_number)]

    # build target circuit
    a,x = processData(ip)
    qc = build_sp_circuit(a,x)
    target = get_state_vector(qc)

    # build approximate circuit
    qc_app, training_params = build_app_circuit(int(np.log2(len(ip))), int(num_layers))
    print(f"\nNumber of parameters to train: {len(training_params)}")

    # link globals for loss functions
    import training_global.losses as L
    L.training_params = training_params
    L.qc_app = qc_app
    L.target = target

    target_sv = Statevector.from_instruction(qc)
    target_amp = target_sv.data
    qc_app_inv = qc_app.inverse()

    import training_reverse.losses as M
    M.qc_app = qc_app
    M.qc = qc
    M.target_sv = target_sv
    M.target_amp = target_amp
    M.qc_app_inv = qc_app_inv

    # random initialization
    params = generate_initial_params(len(training_params), seed=make_seed("initial data generation"))

    # init Adam
    optim = Adam()

    # history (must contain global_fidelity for output_data!)
    history = {
        "params": [],
        "loss": [],
        "global_fidelity": []
    }

    # training loop (keep same behavior: stop only when steps reached)
    for step in range(num_steps):
        local_loss = loss_1q_mean_reverse(params)
        grads = numerical_grad(loss_1q_mean_reverse, params)

        # compute global loss too (MUST record for output_data)
        global_loss = loss_fidelity(params)

        # record
        history["params"].append(params.copy())
        history["loss"].append(local_loss)
        history["global_fidelity"].append(global_loss)

        # print occasionally
        if step % 10 == 0:
            print(f"Step {step}: local={local_loss:.6f}, global={global_loss:.6f}")

        # optimizer update
        params = optim.step(params, grads)

    # final values
    final_loss = history["loss"][-1]
    print("Optimal parameters:", params)
    print("Final local loss:", final_loss)
    print("Final global fidelity loss:", history["global_fidelity"][-1])

    # wrap result
    class DummyResult:
        pass

    result = DummyResult()
    result.x = params
    result.fun = final_loss
    result.history = history

    # save
    output_data(folder_name, data_number, result)

    cont = input("\nTrain another data vector? (y/n): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("\nExiting trainer.")
        break

    print()
