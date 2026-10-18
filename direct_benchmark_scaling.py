"""
Non-interactive HPC benchmark script for scaling experiments.
-------------------------------------------------------------

This version removes all user input. All configuration parameters
are editable at the top of the file. Intended for running on HPC
clusters (SLURM, PBS, etc.) in unattended mode.

It benchmarks:
    - Global fidelity
    - Local mean RDM (k-qubit)
    - Reverse 1q loss
depending on which flags you set.

Original file: benchmark_scaling.py (user-input version)
"""

# ============================================================
# ====================== USER SETTINGS ========================
# ============================================================

# Dataset folder inside data_input/
FOLDER_NAME = "scaling_dataset"

# Qubit range
N_MIN = 3
N_MAX = 6

# Max LBFGSB iterations
MAXITER = 10

# Which methods to run
RUN_GLOBAL = True
RUN_LOCAL_MEAN = True
RUN_REVERSE_1Q = True

# Local RDM settings (only relevant if RUN_LOCAL_MEAN = True)
LOCAL_MEAN_K = 3   # e.g. 1, 2, 3


# ============================================================
# =============== BEGINNING OF ACTUAL PROGRAM ================
# ============================================================

import os
import time
import numpy as np
import math

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


DATA_INPUT_ROOT = "data_input"
DATA_OUTPUT_ROOT = "data_output"


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
        if vec.ndim != 1 or vec.shape[0] != 2**n_qubits:
            raise ValueError(
                f"Vector shape mismatch: expected {2**n_qubits}, got {vec.shape}"
            )
        return vec

    rng = np.random.default_rng(seed=make_seed(f"{folder_name}_data_n{n_qubits}"))
    vec = rng.standard_normal(2**n_qubits)
    np.save(path, vec)
    print(f"  Created new data vector for n={n_qubits}")
    return vec


def auto_layers(n_qubits: int):
    return math.ceil((2**n_qubits) / (2 * n_qubits))


def build_circuits_from_vector(ip, n_qubits, num_layers):
    a, x = processData(ip)
    qc = build_sp_circuit(a, x)
    target = get_state_vector(qc)
    qc_app, training_params = build_app_circuit(n_qubits, num_layers)
    return qc, qc_app, target, training_params


# ============================================================
# 1. GLOBAL FIDELITY BENCHMARK
# ============================================================

def benchmark_global_fidelity(folder_name, n_min, n_max, maxiter):
    print("\n=== Running Global Fidelity Benchmark ===")
    results = []

    for n in range(n_min, n_max + 1):
        num_layers = auto_layers(n)
        num_params = 2 * n * num_layers

        print(f"\n[Global Fidelity] n={n}")
        print(f"  Using {num_layers} layers → {num_params} trainable parameters")

        ip = get_or_create_vector(folder_name, n)
        qc, qc_app, target, training_params = build_circuits_from_vector(ip, n, num_layers)

        L_global.training_params = training_params
        L_global.qc_app = qc_app
        L_global.target = target

        initial = generate_initial_params(len(training_params),
                                          seed=make_seed(f"scale_global_n{n}"))

        t0 = time.time()
        result = optimize_with_lbfgsb(loss_fidelity, initial, maxiter=maxiter)
        t1 = time.time()

        runtime = t1 - t0
        final_loss = float(result.fun)

        print(f"  time = {runtime:.4f} s, final loss = {final_loss:.6e}")
        results.append((n, runtime, final_loss))

    # Save output
    os.makedirs(DATA_OUTPUT_ROOT, exist_ok=True)
    out_file = os.path.join(DATA_OUTPUT_ROOT, f"scaling_global_fidelity_{folder_name}.txt")

    with open(out_file, "w") as f:
        f.write("Benchmark: Global Fidelity\n")
        f.write(f"dataset_folder: {folder_name}\n\n")
        f.write("{:>6} {:>12} {:>16}\n".format("n", "time[s]", "final_loss"))
        f.write("-"*40 + "\n")
        for n, t, loss in results:
            f.write(f"{n:>6d} {t:>12.6f} {loss:>16.6e}\n")

    print(f"Saved → {out_file}")


# ============================================================
# 2. LOCAL MEAN RDM BENCHMARK
# ============================================================

def benchmark_local_mean(folder_name, n_min, n_max, maxiter, k):
    print("\n=== Running Local Mean RDM Benchmark ===")
    print(f"k = {k}")
    results = []

    for n in range(n_min, n_max + 1):
        num_layers = auto_layers(n)
        num_params = 2 * n * num_layers

        print(f"\n[Local Mean RDM] n={n}")
        print(f"  Using {num_layers} layers → {num_params} trainable parameters")

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
        initial = generate_initial_params(len(training_params),
                                          seed=make_seed(f"scale_local_n{n}"))

        t0 = time.time()
        result = train_rdm_lbfgsb(initial, loss_fn=loss_k,
                                  maxiter=maxiter,
                                  label=f"{k}qRDM_n{n}")
        t1 = time.time()

        runtime = t1 - t0
        global_fids = result.history["global_fidelity"]
        best_global_loss = float(np.min(global_fids))

        print(f"  time = {runtime:.4f} s, best global loss = {best_global_loss:.6e}")
        results.append((n, runtime, best_global_loss))

    # Save output
    os.makedirs(DATA_OUTPUT_ROOT, exist_ok=True)
    out_file = os.path.join(DATA_OUTPUT_ROOT, f"scaling_local_mean_k{k}_{folder_name}.txt")

    with open(out_file, "w") as f:
        f.write(f"Benchmark: Local Mean RDM (k={k})\n")
        f.write(f"dataset_folder: {folder_name}\n\n")
        f.write("{:>6} {:>12} {:>18}\n".format("n", "time[s]", "best_global_loss"))
        f.write("-"*45 + "\n")
        for n, t, loss in results:
            f.write(f"{n:>6d} {t:>12.6f} {loss:>18.6e}\n")

    print(f"Saved → {out_file}")


# ============================================================
# 3. REVERSE 1Q BENCHMARK
# ============================================================

def benchmark_reverse_1q(folder_name, n_min, n_max, maxiter):
    print("\n=== Running Reverse 1q Benchmark ===")
    results = []

    for n in range(n_min, n_max + 1):   # IMPORTANT: +1 to include max
        num_layers = auto_layers(n)
        num_params = 2 * n * num_layers

        print(f"\n[Reverse 1q] n={n}")
        print(f"  Using {num_layers} layers → {num_params} trainable parameters")

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

        initial = generate_initial_params(len(training_params),
                                          seed=make_seed(f"scale_reverse_n{n}"))

        t0 = time.time()
        result = optimize_with_lbfgsb(loss_1q_mean_reverse, initial, maxiter=maxiter)
        t1 = time.time()

        runtime = t1 - t0
        final_loss = float(result.fun)

        print(f"  time = {runtime:.4f} s, final loss = {final_loss:.6e}")
        results.append((n, runtime, final_loss))

    # Save output
    os.makedirs(DATA_OUTPUT_ROOT, exist_ok=True)
    out_file = os.path.join(DATA_OUTPUT_ROOT, f"scaling_reverse_1q_{folder_name}.txt")

    with open(out_file, "w") as f:
        f.write("Benchmark: Reverse 1q Loss\n")
        f.write(f"dataset_folder: {folder_name}\n\n")
        f.write("{:>6} {:>12} {:>16}\n".format("n", "time[s]", "final_loss"))
        f.write("-"*40 + "\n")
        for n, t, loss in results:
            f.write(f"{n:>6d} {t:>12.6f} {loss:>16.6e}\n")

    print(f"Saved → {out_file}")


# ============================================================
# MAIN — Auto-run (no user input)
# ============================================================

if __name__ == "__main__":
    print("=== NON-INTERACTIVE SCALING BENCHMARK STARTED ===")
    print(f"Folder = {FOLDER_NAME}")
    print(f"Qubits from {N_MIN} to {N_MAX}")
    print(f"Maxiter = {MAXITER}")
    print()

    if RUN_GLOBAL:
        benchmark_global_fidelity(FOLDER_NAME, N_MIN, N_MAX, MAXITER)

    if RUN_LOCAL_MEAN:
        benchmark_local_mean(FOLDER_NAME, N_MIN, N_MAX, MAXITER, LOCAL_MEAN_K)

    if RUN_REVERSE_1Q:
        benchmark_reverse_1q(FOLDER_NAME, N_MIN, N_MAX, MAXITER)

    print("\n=== BENCHMARK COMPLETED ===")
