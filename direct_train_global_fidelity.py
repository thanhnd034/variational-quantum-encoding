"""
Non-interactive version of Global Fidelity Training
---------------------------------------------------

This script is equivalent to train_global_fidelity.py but:

    • No user input
    • All settings are configured at the top
    • Runs automatically once
    • Fully HPC-friendly for batch execution

Original interactive version: train_global_fidelity.py
"""

# ============================================================
# ====================== USER SETTINGS ========================
# ============================================================

# Dataset folder inside data_input/
FOLDER_NAME = "test1"

# Which data vector index to train (0-based)
DATA_NUMBER = 0

# Number of LBFGSB iterations
MAXITER = 50

# If True → use default_number_of_layers from settings
# If False → override number of layers manually
USE_DEFAULT_LAYERS = True
CUSTOM_NUM_LAYERS = 5   # used only if USE_DEFAULT_LAYERS = False


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
import training_global.losses as L
from utilities.output_data import output_data
import os

print("=== NON-INTERACTIVE GLOBAL FIDELITY TRAINING ===")
print(f"Dataset folder: {FOLDER_NAME}")
print(f"Data index:     {DATA_NUMBER}")
print(f"Maxiter:        {MAXITER}")
print()

# ------------------------------------------------------------
# Load dataset and choose vector
# ------------------------------------------------------------
dataset = load_dataset(FOLDER_NAME)

if DATA_NUMBER < 0 or DATA_NUMBER >= len(dataset):
    raise IndexError(
        f"DATA_NUMBER={DATA_NUMBER} is out of bounds for dataset of size {len(dataset)}"
    )

ip = dataset[DATA_NUMBER]

# ------------------------------------------------------------
# Build circuits
# ------------------------------------------------------------
a, x = processData(ip)
qc = build_sp_circuit(a, x)
target = get_state_vector(qc)

num_qubits = int(np.log2(len(ip)))

if USE_DEFAULT_LAYERS:
    num_layers = default_number_of_layers
else:
    num_layers = CUSTOM_NUM_LAYERS

qc_app, training_params = build_app_circuit(num_qubits, int(num_layers))
print(f"Number of parameters to train: {len(training_params)}")

# Set global-loss module variables
L.training_params = training_params
L.qc_app = qc_app
L.target = target

# ------------------------------------------------------------
# Initial parameters
# ------------------------------------------------------------
initial_params = generate_initial_params(
    len(training_params),
    seed=make_seed("global_fidelity_initial")
)

# ------------------------------------------------------------
# Run LBFGSB optimization
# ------------------------------------------------------------
print("Starting optimization...\n")

result_app = optimize_with_lbfgsb(
    loss_fidelity,
    initial_params,
    maxiter=int(MAXITER)
)

print("\n=== Training Complete ===")
print("Optimal parameters:", result_app.x)
print("Final loss:", result_app.fun)
print()

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
output_data(FOLDER_NAME, DATA_NUMBER, result_app)
print("Output saved using output_data().")

print("\n=== DONE ===")
