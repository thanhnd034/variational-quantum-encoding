"""
Gradient scaling experiment for three loss functions:
    • Global fidelity loss
    • Forward mean 1-qubit RDM loss
    • Reverse mean 1-qubit RDM loss

For each number of qubits n in N_LIST:
    • Build synthetic random input vector of size 2**n
    • Construct exact circuit qc and variational ansatz qc_app
    • Wire up the corresponding loss modules (global, local, reverse)
    • Sample NUM_SAMPLES random parameter vectors
    • Compute ∂L/∂theta_k via the parameter-shift rule (k = PARAM_INDEX)
    • Record |grad| for each sample

Outputs:
    • gradient_scaling.png : plot of mean |grad| vs n (log scale)
    • gradient_scaling_results.npz : arrays with N, means, variances per method

Non-interactive, HPC-friendly.
"""

# ============================================================
# ====================== USER SETTINGS ========================
# ============================================================

# Qubit sizes to test
#N_LIST = [4, 6, 8, 10, 12, 14, 16, 18, 20]
N_LIST = [4, 6, 8]

# Index of parameter to differentiate
PARAM_INDEX = 0

# Number of random samples per n
NUM_SAMPLES = 50

# Enable/disable methods
USE_GLOBAL      = True
USE_FORWARD_RDM = False
USE_REVERSE_RDM = False

# Layer configuration
USE_DEFAULT_LAYERS = False
CUSTOM_NUM_LAYERS = 4

# Master seed
MASTER_SEED = 12345


# ============================================================
# ========================== IMPORTS ==========================
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

from circuits.original import build_sp_circuit
from circuits.approx import build_app_circuit
from circuits.helpers import processData
from utilities.get_state_vector import get_state_vector
from configs.settings import default_number_of_layers

# loss modules
import training_global.losses as L_global
import training_local.losses as L_local
import training_local.train_rdm_mean as Tmean
import training_reverse.losses as L_reverse

from training_global.losses import loss_fidelity
from training_local.train_rdm_mean import make_kq_rdm_loss
from training_reverse.losses import loss_1q_mean_reverse


# ============================================================
# ============== HELPER: PARAMETER-SHIFT GRADIENT ============
# ============================================================

def parameter_shift_grad_1d(loss_fn, params, param_index, shift=np.pi/2):
    """
    Compute gradient using parameter-shift rule.

    dL/dθ = [L(θ + s) - L(θ - s)] / 2   since sin(π/2)=1
    """
    params_plus = np.array(params, dtype=float)
    params_minus = np.array(params, dtype=float)

    params_plus[param_index] += shift
    params_minus[param_index] -= shift

    L_plus = float(loss_fn(params_plus))
    L_minus = float(loss_fn(params_minus))

    return (L_plus - L_minus) / 2.0


# ============================================================
# ============== HELPER: BUILD INSTANCE FOR GIVEN n ==========
# ============================================================

def build_instance_for_n(n, rng):
    """
    For a given number of qubits n:
        • Create synthetic random vector of size 2**n
        • Build exact circuit qc
        • Build variational ansatz qc_app
        • Wire up all loss modules correctly

    Returns
    -------
    num_params : int
        Number of trainable parameters in qc_app.
    forward_1q_loss_fn : callable
        Forward mean 1-qubit RDM loss
    """

    dim = 2**n
    raw_vec = rng.normal(size=dim)

    # preprocess like your pipeline
    a, x = processData(raw_vec)

    # exact circuit
    qc = build_sp_circuit(a, x)
    target_sv = Statevector.from_instruction(qc)
    target_amp = target_sv.data
    target = get_state_vector(qc)

    # ansatz
    if USE_DEFAULT_LAYERS:
        num_layers = default_number_of_layers
    else:
        num_layers = CUSTOM_NUM_LAYERS

    qc_app, training_params = build_app_circuit(n, int(num_layers))
    num_params = len(training_params)

    # ---------------- GLOBAL loss wiring ----------------
    L_global.training_params = training_params
    L_global.qc_app = qc_app
    L_global.target = target

    # ---------------- FORWARD RDM wiring ----------------
    L_local.training_params = training_params
    L_local.qc_app = qc_app
    L_local.target = target
    Tmean.qc_app = qc_app
    forward_1q_loss_fn = make_kq_rdm_loss(1)  # mean 1q RDM loss

    # ---------------- REVERSE RDM wiring ----------------
    qc_app_inv = qc_app.inverse()
    L_reverse.qc_app = qc_app
    L_reverse.qc = qc
    L_reverse.target_sv = target_sv
    L_reverse.target_amp = target_amp
    L_reverse.qc_app_inv = qc_app_inv

    return num_params, forward_1q_loss_fn


# ============================================================
# ========================== MAIN ============================
# ============================================================

def main():
    rng = np.random.default_rng(MASTER_SEED)

    # store results
    N_values = []
    mean_abs_grad_global = []
    var_grad_global = []

    mean_abs_grad_forward = []
    var_grad_forward = []

    mean_abs_grad_reverse = []
    var_grad_reverse = []

    for n in N_LIST:
        print(f"\n=== Processing n = {n} qubits ===")

        num_params, forward_1q_loss_fn = build_instance_for_n(n, rng)

        if PARAM_INDEX >= num_params:
            raise IndexError(
                f"PARAM_INDEX={PARAM_INDEX} out of range for n={n} "
                f"(num_params={num_params})."
            )

        grads_global = []
        grads_forward = []
        grads_reverse = []

        # Monte Carlo sampling over parameter vectors
        for _ in range(NUM_SAMPLES):
            params = rng.uniform(-np.pi, np.pi, size=num_params)

            if USE_GLOBAL:
                g = parameter_shift_grad_1d(loss_fidelity, params, PARAM_INDEX)
                grads_global.append(g)

            if USE_FORWARD_RDM:
                g = parameter_shift_grad_1d(forward_1q_loss_fn, params, PARAM_INDEX)
                grads_forward.append(g)

            if USE_REVERSE_RDM:
                g = parameter_shift_grad_1d(loss_1q_mean_reverse, params, PARAM_INDEX)
                grads_reverse.append(g)

        N_values.append(n)

        # aggregate
        if USE_GLOBAL and grads_global:
            arr = np.abs(np.array(grads_global))
            mean_abs_grad_global.append(np.mean(arr))
            var_grad_global.append(np.var(arr))
            print(f"  Global:  mean|grad|={mean_abs_grad_global[-1]:.3e}")

        else:
            mean_abs_grad_global.append(np.nan)
            var_grad_global.append(np.nan)

        if USE_FORWARD_RDM and grads_forward:
            arr = np.abs(np.array(grads_forward))
            mean_abs_grad_forward.append(np.mean(arr))
            var_grad_forward.append(np.var(arr))
            print(f"  Forward: mean|grad|={mean_abs_grad_forward[-1]:.3e}")

        else:
            mean_abs_grad_forward.append(np.nan)
            var_grad_forward.append(np.nan)

        if USE_REVERSE_RDM and grads_reverse:
            arr = np.abs(np.array(grads_reverse))
            mean_abs_grad_reverse.append(np.mean(arr))
            var_grad_reverse.append(np.var(arr))
            print(f"  Reverse: mean|grad|={mean_abs_grad_reverse[-1]:.3e}")

        else:
            mean_abs_grad_reverse.append(np.nan)
            var_grad_reverse.append(np.nan)

    # convert to arrays
    N_values = np.array(N_values, dtype=int)
    mean_abs_grad_global = np.array(mean_abs_grad_global)
    mean_abs_grad_forward = np.array(mean_abs_grad_forward)
    mean_abs_grad_reverse = np.array(mean_abs_grad_reverse)

    # ========================================================
    # ====================== SAVE DATA =======================
    # ========================================================

    np.savez(
        "gradient_scaling_results.npz",
        N=N_values,
        mean_abs_grad_global=mean_abs_grad_global,
        mean_abs_grad_forward=mean_abs_grad_forward,
        mean_abs_grad_reverse=mean_abs_grad_reverse,
        PARAM_INDEX=PARAM_INDEX,
        NUM_SAMPLES=NUM_SAMPLES,
    )
    print("\nSaved data to gradient_scaling_results.npz")


# ========================================================
# =================== SAVE TEXT SUMMARY ===================
# ========================================================

    with open("gradient_scaling_summary.txt", "w") as f:
        f.write("Gradient Scaling Summary\n")
        f.write("========================\n\n")
        f.write(f"Parameter index: {PARAM_INDEX}\n")
        f.write(f"Samples per n:   {NUM_SAMPLES}\n\n")

        f.write("n\tGlobal\tForwardRDM\tReverseRDM\n")
        for i, n in enumerate(N_values):
            g  = mean_abs_grad_global[i]  if USE_GLOBAL      else np.nan
            f1 = mean_abs_grad_forward[i] if USE_FORWARD_RDM else np.nan
            r1 = mean_abs_grad_reverse[i] if USE_REVERSE_RDM else np.nan

            f.write(f"{n}\t{g:.6e}\t{f1:.6e}\t{r1:.6e}\n")

    print("Saved gradient summary to gradient_scaling_summary.txt")


    # ========================================================
    # ====================== MAKE PLOT =======================
    # ========================================================

    plt.figure(figsize=(7, 5))

    if USE_GLOBAL:
        plt.semilogy(N_values, mean_abs_grad_global, "o-", label="Global fidelity")

    if USE_FORWARD_RDM:
        plt.semilogy(N_values, mean_abs_grad_forward, "s-", label="Forward RDM (1q mean)")

    if USE_REVERSE_RDM:
        plt.semilogy(N_values, mean_abs_grad_reverse, "d-", label="Reverse RDM (1q mean)")

    plt.xlabel("Number of qubits $n$")
    plt.ylabel(r"Mean $|\partial L / \partial \theta_{%d}|$" % (PARAM_INDEX + 1))
    plt.title("Gradient scaling with system size")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gradient_scaling.png", dpi=300)

    print("Saved plot to gradient_scaling.png")


if __name__ == "__main__":
    main()
