"""
Non-interactive version of Reverse 1Q Local Loss Training
----------------------------------------------------------

This script is the HPC-ready equivalent of train_reverse_local_loss.py.

It:

    • Loads dataset without asking the user
    • Selects which data vector to train
    • Builds the forward qc and qc_app circuits
    • Constructs the reverse 1q mean RDM loss
    • Runs LBFGSB optimization
    • Saves results with output_data()
    • Is fully non-interactive (no input prompts)
    • Suitable for SLURM or other HPC batch systems

Original interactive file: train_reverse_local_loss.py
"""

# ============================================================
# ====================== USER SETTINGS ========================
# ============================================================

# Dataset folder inside data_input/
FOLDER_NAME = "test1"

# Index of the data vector to train (0-based)
DATA_NUMBER = 0

# Number of LBFGSB iterations
MAXITER = 30

# Whether to use the default number_of_layers from configs.settings
USE_DEFAULT_LAYERS = True
CUSTOM_NUM_LAYERS = 5     # only used if USE_DEFAULT_LAYERS = False


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


print("=== NON-INTERACTIVE REVERSE 1Q LOCAL LOSS TRAINING ===")
print(f"Dataset folder:  {FOLDER_NAME}")
print(f"Data index:      {DATA_NUMBER}")
print(f"Maxiter:         {MAXITER}")
print()


# ------------------------------------------------------------
# Load dataset + extract selected input vector
# ------------------------------------------------------------
dataset = load_dataset(FOLDER_NAME)

if DATA_NUMBER < 0 or DATA_NUMBER >= len(dataset):
    raise IndexError(
        f"DATA_NUMBER={DATA_NUMBER} is invalid for dataset of size {len(dataset)}"
    )

ip = dataset[DATA_NUMBER]


# ------------------------------------------------------------
# Build forward circuit qc from data
# ------------------------------------------------------------
a, x = processData(ip)
qc = build_sp_circuit(a, x)
target_sv = Statevector.from_instruction(qc)
target_amp = target_sv.data


# ------------------------------------------------------------
# Build variational circuit qc_app
# ------------------------------------------------------------
num_qubits = int(np.log2(len(ip)))

if USE_DEFAULT_LAYERS:
    num_layers = default_number_of_layers
else:
    num_layers = CUSTOM_NUM_LAYERS

qc_app, training_params = build_app_circuit(num_qubits, int(num_layers))
qc_app_inv = qc_app.inverse()

print(f"Number of parameters to train: {len(training_params)}")
print()


# ------------------------------------------------------------
# Attach references to global loss module (if needed)
# ------------------------------------------------------------
L_global.training_params = training_params
L_global.qc_app = qc_app
L_global.target = get_state_vector(qc)


# ------------------------------------------------------------
# Attach references to reverse-loss module
# ------------------------------------------------------------
L_reverse.qc_app = qc_app
L_reverse.qc = qc
L_reverse.target_sv = target_sv
L_reverse.target_amp = target_amp
L_reverse.qc_app_inv = qc_app_inv

from training_reverse.losses import loss_1q_mean_reverse


# ------------------------------------------------------------
# Generate initial parameters for optimization
# ------------------------------------------------------------
initial_params = generate_initial_params(
    len(training_params),
    seed=make_seed("reverse_1q_initial")
)


# ------------------------------------------------------------
# Run LBFGSB optimization
# ------------------------------------------------------------
print("Starting LBFGSB optimization...\n")

result_app = optimize_with_lbfgsb(
    loss_1q_mean_reverse,
    initial_params,
    maxiter=int(MAXITER)
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
print("\n=== Training Complete ===")
print("Optimal parameters:", result_app.x)
print("Final loss:", result_app.fun)


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
output_data(FOLDER_NAME, DATA_NUMBER, result_app)

print("\nOutput saved using output_data().")
print("\n=== DONE ===")
