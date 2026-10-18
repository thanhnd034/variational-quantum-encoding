"""
Non-interactive version of MULTI-VECTOR Reverse 1Q Local Loss Training
----------------------------------------------------------------------

This script is the HPC-ready equivalent of train_multiple_reverse.py.

It:

    • Loads dataset once
    • Trains reverse 1q local loss on a RANGE of data vectors
    • Has NO USER INPUT (all settings at the top)
    • Uses the same loss function and output format
    • Is ideal for SLURM batch jobs: run multiple vectors in one job

Original interactive file: train_multiple_reverse.py
"""

# ============================================================
# ====================== USER SETTINGS ========================
# ============================================================

# Dataset folder under data_input/
FOLDER_NAME = "test1"

# Train for data indices [I_START ... I_END] inclusive
I_START = 0
I_END   = 1

# Number of LBFGSB iterations
MAXITER = 20

# Whether to use default number of layers from settings
USE_DEFAULT_LAYERS = True
CUSTOM_NUM_LAYERS  = 5   # used only if USE_DEFAULT_LAYERS = False


# ============================================================
# =============== BEGINNING OF ACTUAL PROGRAM ================
# ============================================================

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

import training_global.losses as L_global
import training_reverse.losses as L_reverse
from training_reverse.losses import loss_1q_mean_reverse


print("=== NON-INTERACTIVE MULTI-VECTOR REVERSE 1Q TRAINING ===")
print(f"Dataset folder: {FOLDER_NAME}")
print(f"Training vectors: {I_START} → {I_END}")
print(f"Maxiter: {MAXITER}")
print()


# ------------------------------------------------------------
# Load entire dataset
# ------------------------------------------------------------
dataset = load_dataset(FOLDER_NAME)
dataset_size = len(dataset)

# Safety checks
if I_START < 0 or I_END >= dataset_size or I_START > I_END:
    raise IndexError(
        f"Invalid index range: I_START={I_START}, I_END={I_END}, dataset_size={dataset_size}"
    )


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
for data_number in range(I_START, I_END + 1):

    print(f"\n=== Training data vector #{data_number} ===")

    ip = dataset[data_number]

    # build the original circuit qc
    a, x = processData(ip)
    qc = build_sp_circuit(a, x)
    target = get_state_vector(qc)

    # build qc_app
    n_qubits = int(np.log2(len(ip)))

    if USE_DEFAULT_LAYERS:
        num_layers = default_number_of_layers
    else:
        num_layers = CUSTOM_NUM_LAYERS

    qc_app, training_params = build_app_circuit(n_qubits, int(num_layers))
    print(f"Number of parameters to train: {len(training_params)}")

    # Global fidelity module setup
    L_global.training_params = training_params
    L_global.qc_app = qc_app
    L_global.target = target

    # Reverse-specific static values
    target_sv = Statevector.from_instruction(qc)
    target_amp = target_sv.data
    qc_app_inv = qc_app.inverse()

    # Setup reverse-loss globals
    L_reverse.qc_app = qc_app
    L_reverse.qc = qc
    L_reverse.target_sv = target_sv
    L_reverse.target_amp = target_amp
    L_reverse.qc_app_inv = qc_app_inv

    # Generate initial parameters
    initial_params = generate_initial_params(
        len(training_params),
        seed=make_seed(f"initial_vec_{data_number}")
    )

    # Perform optimization
    print("  Starting LBFGSB optimization...")

    result_app = optimize_with_lbfgsb(
        loss_1q_mean_reverse,
        initial_params,
        maxiter=int(MAXITER)
    )

    print(f"  Final loss: {result_app.fun:.6e}")
    print("  Saving results...")

    # Store everything
    output_data(FOLDER_NAME, data_number, result_app)

print("\n=== ALL TRAINING FINISHED SUCCESSFULLY ===")
