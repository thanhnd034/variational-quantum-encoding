"""
Non-interactive version of Local Mean RDM Training
--------------------------------------------------

This script is the non-interactive equivalent of train_local_mean_rdm.py.
It:

    • Loads dataset automatically
    • Selects which vector to train
    • Selects k for k-qubit mean RDM loss
    • Performs LBFGSB optimization
    • Saves outputs with output_data()
    • Does NOT ask the user for any input
    • Is safe for batch execution on HPC clusters

Original interactive file: train_local_mean_rdm.py
"""

# ============================================================
# ====================== USER SETTINGS ========================
# ============================================================

# Dataset folder inside data_input/
FOLDER_NAME = "test1"

# Index of the data vector to train (0-based)
DATA_NUMBER = 0

# Number of LBFGSB iterations
MAXITER = 300

# Number of qubits used in k-qubit mean RDM (k = 1,2,3,...)
K_RDM = 2

# Whether to use default number of layers from settings
USE_DEFAULT_LAYERS = False
CUSTOM_NUM_LAYERS = 4   # used only if USE_DEFAULT_LAYERS=False


# ============================================================
# =============== BEGINNING OF ACTUAL PROGRAM ================
# ============================================================

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
from utilities.output_data import output_data
from training_local.train_rdm_mean import make_kq_rdm_loss, train_rdm_lbfgsb

import training_global.losses as L_global
import training_local.losses as L_local
import training_local.train_rdm_mean as Tmean


print("=== NON-INTERACTIVE LOCAL MEAN RDM TRAINING ===")
print(f"Dataset folder:      {FOLDER_NAME}")
print(f"Data vector index:   {DATA_NUMBER}")
print(f"Maxiter:             {MAXITER}")
print(f"k-qubit RDM:         {K_RDM}")
print()

# ------------------------------------------------------------
# Load dataset and prepare input vector
# ------------------------------------------------------------
dataset = load_dataset(FOLDER_NAME)

if DATA_NUMBER < 0 or DATA_NUMBER >= len(dataset):
    raise IndexError(
        f"DATA_NUMBER={DATA_NUMBER} is invalid for dataset size {len(dataset)}"
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

print(f"Number of parameters to train: {len(training_params)}\n")

# ------------------------------------------------------------
# Attach parameters to global and local loss modules
# ------------------------------------------------------------
L_global.training_params = training_params
L_global.qc_app = qc_app
L_global.target = target

L_local.training_params = training_params
L_local.qc_app = qc_app
L_local.target = target

Tmean.qc_app = qc_app  # required by train_rdm_lbfgsb

# ------------------------------------------------------------
# Prepare loss function
# ------------------------------------------------------------
loss_k = make_kq_rdm_loss(int(K_RDM))

# ------------------------------------------------------------
# Initial parameters
# ------------------------------------------------------------
initial_params = generate_initial_params(
    len(training_params),
    seed=make_seed("local_rdm_initial")
)

# ------------------------------------------------------------
# Run optimization
# ------------------------------------------------------------
print("Starting LBFGSB optimization...\n")

result_app = train_rdm_lbfgsb(
    initial_params,
    loss_fn=loss_k,
    maxiter=int(MAXITER),
    label=f"{K_RDM}q mean RDM"
)

print("\n=== Training Complete ===")

# ------------------------------------------------------------
# Retrieve best cycle info
# ------------------------------------------------------------
global_fids = result_app.history.get("global_fidelity", [])

if isinstance(global_fids, list) and len(global_fids) > 0:
    best_idx = int(np.argmin(global_fids))
    best_cycle  = best_idx + 1
    best_loss   = global_fids[best_idx]
    best_params = result_app.history["params"][best_idx]

    print(f"Best cycle:       {best_cycle}")
    print(f"Best global_loss: {best_loss}")
else:
    print("WARNING: global_fidelity history is empty — no best cycle computed.")
    best_cycle = None

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
output_data(FOLDER_NAME, DATA_NUMBER, result_app)

print("\nOutput saved using output_data().")
print("\n=== DONE ===")
