"""
Non-interactive HPC benchmark script
------------------------------------

All editable settings are at the top of this file.
No user input required. Designed for batch jobs on HPC systems.

This script benchmarks:
    - Global fidelity (optional)
    - Local mean RDM (optional)
    - Reverse 1q local loss (optional)

Early stopping is enabled using plateau loss.

"""

# ============================================================
# ====================== USER SETTINGS ========================
# ============================================================

# Folder under data_input/ to store or load vectors
FOLDER_NAME = "benchmark_dataset"

# Range of qubits
N_MIN = 2
N_MAX = 6

# Max L-BFGS-B iterations
MAXITER = 500

# Universal convergence threshold (imported from settings, but can override here)
USE_CUSTOM_THRESHOLD = True
CUSTOM_TARGET_THRESHOLD = 1e-3   # Only used if USE_CUSTOM_THRESHOLD = True

# Which methods to run
RUN_GLOBAL = True
RUN_LOCAL_MEAN = False
RUN_REVERSE_1Q = True

# Local RDM settings (only used if RUN_LOCAL_MEAN=True)
LOCAL_MEAN_K = 1   # e.g. 1, 2, or 3


# ============================================================
# =============== BEGINNING OF ACTUAL PROGRAM ================
# ============================================================

import os
import time
import math
import numpy as np

from circuits.helpers import processData
from circuits.approx import build_app_circuit
from circuits.original import build_sp_circuit
from utilities.get_state_vector import get_state_vector
from utilities.initial_params import generate_initial_params
from utilities.make_seed import make_seed

from training_global.losses import loss_fidelity
import training_global.losses as L_global

from training_local.train_rdm_mean import make_kq_rdm_loss, train_rdm_lbfgsb
import training_local.losses as L_local

import training_reverse.losses as L_rev
from training_reverse.losses import loss_1q_mean_reverse

from maths.optimize_with_lbfgsb import optimize_with_lbfgsb
from qiskit.quantum_info import Statevector

# Import global Default threshold
from configs.settings import TARGET_THRESHOLD

# If user wants to override the threshold locally for this batch job
if USE_CUSTOM_THRESHOLD:
    TARGET_THRESHOLD = CUSTOM_TARGET_THRESHOLD


DATA_INPUT_ROOT = "data_input"
DATA_OUTPUT_ROOT = "data_output"


# ============================================================
# Plateau-loss wrapper (no optimizer modification needed)
# ============================================================

def make_plateau_loss(original_loss_fn, threshold):
    state = {"triggered": False}

    def wrapped_loss(params):
        if state["triggered"]:
            return threshold

        value = original_loss_fn(params)
        if value <= threshold:
            state["triggered"] = True
            return threshold

        return value

    return wrapped_loss


# ============================================================
# Utilities
# ============================================================

def get_or_create_vector(folder_name: str, n_qubits: int) -> np.ndarray:
    folder_path = os.path.join(DATA_INPUT_ROOT, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    filename = f"n{n_qubits}_vec0.npy"
    path = os.path.join(folder_path, filename)

    if os.path.exists(path):
        vec = np.load(path)
        return vec

    rng = np.random.default_rng(seed=make_seed(f"{folder_name}_data_n{n_qubits}"))
    vec = rng.standard_normal(2 ** n_qubits)
    np.save(path, vec)
    print(f"  Created new random vector for n={n_qubits}")
    return vec


def auto_layers(n_qubits: int) -> int:
    return math.ceil((2**n_qubits) / (2 * n_qubits))


def build_circuits_from_vector(ip: np.ndarray, n_qubits: int, num_layers: int):
    a, x = processData(ip)
    qc = build_sp_circuit(a, x)
    target = get_state_vector(qc)
    qc_app, training_params = build_app_circuit(n_qubits, num_layers)
    return qc, qc_app, target, training_params


# ============================================================
# Time-to-threshold extractor
# ============================================================

def compute_time_to_threshold(runtime: float, history):
    if not isinstance(history, dict):
        return False, None, None
    losses = history.get("loss", [])
    if not losses:
        return False, None, None

    idx = None
    for i, val in enumerate(losses):
        if val <= TARGET_THRESHOLD:
            idx = i + 1
            break

    if idx is None:
        return False, None, None

    frac = idx / len(losses)
    return True, runtime * frac, idx


# ============================================================
# 1. GLOBAL FIDELITY BENCHMARK
# ============================================================

def benchmark_global(folder_name, n_min, n_max, maxiter):
    results = []

    print("\n=== Running Global Fidelity Benchmark ===")
    print(f"Threshold = {TARGET_THRESHOLD}\n")

    for n in range(n_min, n_max + 1):
        num_layers = auto_layers(n)
        num_params = 2 * n * num_layers

        print(f"\n[Global] n={n}")
        print(f"  Using {num_layers} layers → {num_params} parameters")

        ip = get_or_create_vector(folder_name, n)
        qc, qc_app, target, training_params = build_circuits_from_vector(ip, n, num_layers)

        L_global.training_params = training_params
        L_global.qc_app = qc_app
        L_global.target = target

        initial = generate_initial_params(len(training_params),
                                          seed=make_seed(f"global_n{n}"))

        wrapped_loss = make_plateau_loss(loss_fidelity, TARGET_THRESHOLD)

        t0 = time.time()
        result = optimize_with_lbfgsb(
            wrapped_loss,
            initial,
            global_fidelity_fn=loss_fidelity,
            maxiter=int(maxiter),
        )
        t1 = time.time()

        runtime = t1 - t0
        final_loss = float(result.fun)
        history = result.history

        reached, t_thr, it_thr = compute_time_to_threshold(runtime, history)

        results.append((n, runtime, final_loss, t_thr, it_thr, reached))

    out = os.path.join(DATA_OUTPUT_ROOT,
                       f"global_threshold_{folder_name}.txt")
    os.makedirs(DATA_OUTPUT_ROOT, exist_ok=True)

    with open(out, "w") as f:
        f.write("Global Fidelity Benchmark (plateau early-stop)\n")
        f.write(f"Threshold = {TARGET_THRESHOLD}\n\n")
        f.write("{:>4} {:>12} {:>12} {:>14} {:>10} {:>8}\n"
                .format("n", "time[s]", "fin_loss", "t_thr[s]", "it_thr", "reach"))
        f.write("-"*70 + "\n")

        for n, t, fl, tt, it, r in results:
            f.write("{:>4d} {:>12.4f} {:>12.3e} {:>14s} {:>10s} {:>8s}\n".format(
                n, t, fl,
                f"{tt:.4f}" if tt is not None else "NA",
                f"{it}" if it is not None else "NA",
                "1" if r else "0"
            ))

    print(f"Saved → {out}")


# ============================================================
# 2. LOCAL MEAN RDM BENCHMARK
# ============================================================

def benchmark_local(folder_name, n_min, n_max, maxiter, k):
    results = []

    print("\n=== Running Local Mean RDM Benchmark ===")
    print(f"k = {k}, Threshold = {TARGET_THRESHOLD}\n")

    for n in range(n_min, n_max + 1):
        num_layers = auto_layers(n)
        num_params = 2 * n * num_layers

        print(f"\n[Local Mean] n={n}")
        print(f"  Using {num_layers} layers → {num_params} parameters")

        ip = get_or_create_vector(folder_name, n)
        qc, qc_app, target, training_params = build_circuits_from_vector(ip, n, num_layers)

        L_local.training_params = training_params
        L_local.qc_app = qc_app
        L_local.target = target

        L_global.training_params = training_params
        L_global.qc_app = qc_app
        L_global.target = target

        import training_local.train_rdm_mean as Tmean
        Tmean.qc_app = qc_app

        loss_k = make_kq_rdm_loss(k)
        wrapped_loss = make_plateau_loss(loss_k, TARGET_THRESHOLD)

        initial = generate_initial_params(len(training_params),
                                          seed=make_seed(f"local_n{n}"))

        t0 = time.time()
        result = train_rdm_lbfgsb(
            initial,
            loss_fn=wrapped_loss,
            maxiter=int(maxiter),
            label=f"{k}qRDM_n{n}"
        )
        t1 = time.time()

        runtime = t1 - t0
        hist = result.history

        best_gl = (float(np.min(hist["global_fidelity"]))
                   if "global_fidelity" in hist and hist["global_fidelity"]
                   else None)

        reached, t_thr, it_thr = compute_time_to_threshold(runtime, hist)

        results.append((n, runtime, best_gl, t_thr, it_thr, reached))

    out = os.path.join(DATA_OUTPUT_ROOT,
                       f"local_mean_k{k}_{folder_name}.txt")
    with open(out, "w") as f:
        f.write(f"Local Mean RDM Benchmark (k={k})\n")
        f.write(f"Threshold = {TARGET_THRESHOLD}\n\n")
        f.write("{:>4} {:>12} {:>18} {:>14} {:>10} {:>8}\n"
                .format("n", "time[s]", "best_gl", "t_thr[s]", "it_thr", "reach"))
        f.write("-"*90 + "\n")

        for n, T, bg, tt, it, r in results:
            f.write("{:>4d} {:>12.4f} {:>18s} {:>14s} {:>10s} {:>8s}\n".format(
                n, T,
                f"{bg:.3e}" if bg is not None else "NA".rjust(18),
                f"{tt:.4f}" if tt else "NA",
                f"{it}" if it else "NA",
                "1" if r else "0"
            ))

    print(f"Saved → {out}")


# ============================================================
# 3. REVERSE 1Q BENCHMARK
# ============================================================

def benchmark_reverse(folder_name, n_min, n_max, maxiter):
    results = []

    print("\n=== Running Reverse 1q Benchmark ===")
    print(f"Threshold = {TARGET_THRESHOLD}\n")

    for n in range(n_min, n_max + 1):
        num_layers = auto_layers(n)
        num_params = 2 * n * num_layers

        print(f"\n[Reverse 1q] n={n}")
        print(f"  Using {num_layers} layers → {num_params} parameters")

        ip = get_or_create_vector(folder_name, n)
        qc, qc_app, target, training_params = build_circuits_from_vector(ip, n, num_layers)

        L_global.training_params = training_params
        L_global.qc_app = qc_app
        L_global.target = target

        target_sv = Statevector.from_instruction(qc)
        qc_app_inv = qc_app.inverse()

        L_rev.qc_app = qc_app
        L_rev.qc = qc
        L_rev.target_sv = target_sv
        L_rev.target_amp = target_sv.data
        L_rev.qc_app_inv = qc_app_inv

        wrapped_loss = make_plateau_loss(loss_1q_mean_reverse, TARGET_THRESHOLD)

        initial = generate_initial_params(len(training_params),
                                          seed=make_seed(f"reverse_n{n}"))

        t0 = time.time()
        result = optimize_with_lbfgsb(
            wrapped_loss,
            initial,
            global_fidelity_fn=loss_fidelity,
            maxiter=int(maxiter),
        )
        t1 = time.time()

        runtime = t1 - t0
        final_loss = float(result.fun)
        hist = result.history
        reached, t_thr, it_thr = compute_time_to_threshold(runtime, hist)

        results.append((n, runtime, final_loss, t_thr, it_thr, reached))

    out = os.path.join(DATA_OUTPUT_ROOT,
                       f"reverse_1q_{folder_name}.txt")
    with open(out, "w") as f:
        f.write("Reverse 1q Benchmark\n")
        f.write(f"Threshold = {TARGET_THRESHOLD}\n\n")
        f.write("{:>4} {:>12} {:>12} {:>14} {:>10} {:>8}\n"
                .format("n", "time[s]", "fin_loss", "t_thr[s]", "it_thr", "reach"))
        f.write("-"*80 + "\n")

        for n, T, fl, tt, it, r in results:
            f.write("{:>4d} {:>12.4f} {:>12.3e} {:>14s} {:>10s} {:>8s}\n".format(
                n, T, fl,
                f"{tt:.4f}" if tt else "NA",
                f"{it}" if it else "NA",
                "1" if r else "0"
            ))

    print(f"Saved → {out}")


# ============================================================
# MAIN (AUTO-RUN)
# ============================================================

if __name__ == "__main__":

    print("=== NON-INTERACTIVE HPC BENCHMARK STARTED ===")
    print(f"Folder = {FOLDER_NAME}")
    print(f"Qubits from {N_MIN} to {N_MAX}")
    print(f"Maxiter = {MAXITER}")
    print(f"Threshold = {TARGET_THRESHOLD}\n")

    if RUN_GLOBAL:
        benchmark_global(FOLDER_NAME, N_MIN, N_MAX, MAXITER)

    if RUN_LOCAL_MEAN:
        benchmark_local(FOLDER_NAME, N_MIN, N_MAX, MAXITER, LOCAL_MEAN_K)

    if RUN_REVERSE_1Q:
        benchmark_reverse(FOLDER_NAME, N_MIN, N_MAX, MAXITER)

    print("\n=== BENCHMARK COMPLETED ===")
