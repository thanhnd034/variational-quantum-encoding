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

from configs.settings import default_benchmark_min, default_benchmark_max, TARGET_THRESHOLD


DATA_INPUT_ROOT = "data_input"
DATA_OUTPUT_ROOT = "data_output"

DEFAULT_N_MIN = default_benchmark_min
DEFAULT_N_MAX = default_benchmark_max


# ============================================================
# Plateau-loss wrapper (NO change to optimizer required)
# ============================================================

def make_plateau_loss(original_loss_fn, threshold):
    """
    Once original_loss_fn(params) drops below the threshold, we freeze
    the landscape by always returning 'threshold'. The optimizer will
    naturally converge early because no improvement is possible.

    This avoids modifying optimize_with_lbfgsb.py completely.
    """
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
# Utility: Get or create data vectors
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
                f"Existing file {path} has shape {vec.shape}, expected length {2**n_qubits}"
            )
        return vec

    rng = np.random.default_rng(seed=make_seed(f"{folder_name}_data_n{n_qubits}"))
    vec = rng.standard_normal(2 ** n_qubits)

    np.save(path, vec)
    print(f"  Created new data vector for n={n_qubits} at {path}")
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
# Helper: compute time-to-threshold from stored history
# ============================================================

def _compute_time_to_threshold(runtime: float, history):
    if not isinstance(history, dict):
        return False, None, None
    if "loss" not in history:
        return False, None, None

    losses = history["loss"]
    if not losses:
        return False, None, None

    # find the first iteration <= threshold
    trigger_idx = None
    for i, val in enumerate(losses):
        if val <= TARGET_THRESHOLD:
            trigger_idx = i + 1
            break

    if trigger_idx is None:
        return False, None, None

    # proportional estimate of time
    frac = trigger_idx / len(losses)
    return True, runtime * frac, trigger_idx


# ============================================================
# 1. GLOBAL FIDELITY BENCHMARK
# ============================================================

def benchmark_global_fidelity(folder_name: str, n_min: int, n_max: int, maxiter: int):
    results = []

    for n in range(n_min, n_max + 1):
        num_layers = auto_layers(n)
        num_params = 2 * n * num_layers

        print(f"\n[Global Fidelity] n = {n} qubits")
        print(f"  Using {num_layers} layers → {num_params} parameters")
        print(f"  Threshold = {TARGET_THRESHOLD:g}")

        ip = get_or_create_vector(folder_name, n)
        qc, qc_app, target, training_params = build_circuits_from_vector(
            ip, n, num_layers
        )

        L_global.training_params = training_params
        L_global.qc_app = qc_app
        L_global.target = target

        initial_params = generate_initial_params(
            len(training_params), seed=make_seed(f"bench_n{n}")
        )

        # ---> Wrap loss to enforce early stopping
        wrapped_loss = make_plateau_loss(loss_fidelity, TARGET_THRESHOLD)

        t0 = time.time()
        result_app = optimize_with_lbfgsb(
            wrapped_loss,
            initial_params,
            global_fidelity_fn=loss_fidelity,
            maxiter=int(maxiter),
        )
        t1 = time.time()

        runtime = t1 - t0
        final_loss = float(result_app.fun)
        history = result_app.history

        reached, t_thr, iter_thr = _compute_time_to_threshold(runtime, history)

        if reached:
            print(
                f"  time_total = {runtime:.4f}s, final_loss = {final_loss:.3e}, "
                f"time_to_thr = {t_thr:.4f}s at iter {iter_thr}"
            )
        else:
            print(
                f"  time_total = {runtime:.4f}s, final_loss = {final_loss:.3e}, "
                f"did NOT reach threshold."
            )

        results.append((n, runtime, final_loss, t_thr, iter_thr, reached))

    out_path = os.path.join(DATA_OUTPUT_ROOT, f"benchmark_global_fidelity_{folder_name}.txt")
    os.makedirs(DATA_OUTPUT_ROOT, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("Benchmark: Global Fidelity (Early-stop using plateau loss)\n")
        f.write(f"Threshold = {TARGET_THRESHOLD:g}\n\n")
        f.write("{:>4s} {:>12s} {:>12s} {:>14s} {:>10s} {:>8s}\n"
                .format("n", "time[s]", "final_loss", "time_thr[s]", "iter_thr", "reached"))
        f.write("-"*70 + "\n")

        for n, t, fl, tthr, ithr, r in results:
            f.write("{:>4d} {:>12.5f} {:>12.3e} {:>14s} {:>10s} {:>8s}\n".format(
                n,
                t,
                fl,
                f"{tthr:.5f}" if tthr is not None else "NA",
                f"{ithr}" if ithr is not None else "NA",
                "1" if r else "0"
            ))

    print(f"\nSaved to {out_path}")


# ============================================================
# 2. LOCAL MEAN RDM BENCHMARK
# ============================================================

def benchmark_local_mean(folder_name, n_min, n_max, maxiter, k):
    results = []

    for n in range(n_min, n_max + 1):
        num_layers = auto_layers(n)
        num_params = 2 * n * num_layers
        print(f"\n[Local Mean RDM] k={k}, n={n}")
        print(f"  Using {num_layers} layers → {num_params} parameters")
        print(f"  Threshold = {TARGET_THRESHOLD:g}")

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

        loss_k_mean = make_kq_rdm_loss(k)
        wrapped_loss = make_plateau_loss(loss_k_mean, TARGET_THRESHOLD)

        initial_params = generate_initial_params(len(training_params),
                                                 seed=make_seed(f"bench_n{n}"))

        t0 = time.time()
        result_app = train_rdm_lbfgsb(
            initial_params,
            loss_fn=wrapped_loss,
            maxiter=int(maxiter),
            label=f"{k}q mean RDM benchmark n={n}",
        )
        t1 = time.time()

        runtime = t1 - t0
        hist = result_app.history

        if "global_fidelity" in hist and len(hist["global_fidelity"]) > 0:
            best_global_loss = float(np.min(hist["global_fidelity"]))
        else:
            best_global_loss = None

        reached, t_thr, iter_thr = _compute_time_to_threshold(runtime, hist)

        results.append((n, runtime, best_global_loss, t_thr, iter_thr, reached))

    out_path = os.path.join(DATA_OUTPUT_ROOT,
                            f"benchmark_local_mean_k{k}_{folder_name}.txt")
    os.makedirs(DATA_OUTPUT_ROOT, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"Benchmark: Local Mean RDM (k={k}) with plateau-loss early stop\n")
        f.write(f"Threshold = {TARGET_THRESHOLD:g}\n\n")
        f.write("{:>4s} {:>12s} {:>18s} {:>14s} {:>10s} {:>8s}\n"
                .format("n", "time[s]", "best_global", "time_thr[s]", "iter_thr", "reached"))
        f.write("-"*80 + "\n")

        for n, t, bg, tthr, ithr, r in results:
            f.write("{:>4d} {:>12.5f} {:>18s} {:>14s} {:>10s} {:>8s}\n".format(
                n,
                t,
                f"{bg:.3e}" if bg is not None else "NA".rjust(18),
                f"{tthr:.5f}" if tthr is not None else "NA",
                f"{ithr}" if ithr is not None else "NA",
                "1" if r else "0",
            ))

    print(f"\nSaved to {out_path}")


# ============================================================
# 3. REVERSE 1Q BENCHMARK
# ============================================================

def benchmark_reverse_1q(folder_name, n_min, n_max, maxiter):
    results = []

    for n in range(n_min, n_max + 1):
        num_layers = auto_layers(n)
        num_params = 2 * n * num_layers
        print(f"\n[Reverse 1q Local] n={n}")
        print(f"  Using {num_layers} layers → {num_params} parameters")
        print(f"  Threshold = {TARGET_THRESHOLD:g}")

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

        initial_params = generate_initial_params(len(training_params),
                                                 seed=make_seed(f"bench_n{n}"))

        t0 = time.time()
        result_app = optimize_with_lbfgsb(
            wrapped_loss,
            initial_params,
            global_fidelity_fn=loss_fidelity,
            maxiter=int(maxiter),
        )
        t1 = time.time()

        runtime = t1 - t0
        final_loss = float(result_app.fun)
        hist = result_app.history
        reached, t_thr, iter_thr = _compute_time_to_threshold(runtime, hist)

        results.append((n, runtime, final_loss, t_thr, iter_thr, reached))

    out_path = os.path.join(DATA_OUTPUT_ROOT,
                            f"benchmark_reverse_1q_{folder_name}.txt")
    os.makedirs(DATA_OUTPUT_ROOT, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("Benchmark: Reverse 1q (LBFGSB, plateau-loss early stop)\n")
        f.write(f"Threshold = {TARGET_THRESHOLD:g}\n\n")
        f.write("{:>4s} {:>12s} {:>12s} {:>14s} {:>10s} {:>8s}\n"
                .format("n", "time[s]", "final_loss", "time_thr[s]", "iter_thr", "reached"))
        f.write("-"*75 + "\n")

        for n, t, fl, tthr, ithr, r in results:
            f.write("{:>4d} {:>12.5f} {:>12.3e} {:>14s} {:>10s} {:>8s}\n".format(
                n,
                t,
                fl,
                f"{tthr:.5f}" if tthr is not None else "NA",
                f"{ithr}" if ithr is not None else "NA",
                "1" if r else "0",
            ))

    print(f"\nSaved to {out_path}")


# ============================================================
# MAIN ENTRY
# ============================================================

def main():
    print("=== Benchmark Scaling (Plateau-loss early stopping enabled) ===\n")
    print(f"Universal threshold = {TARGET_THRESHOLD:g}\n")

    folder_name = input("Dataset folder name: ").strip()

    n_min = int(input(f"Minimum n_qubits [{DEFAULT_N_MIN}]: ") or DEFAULT_N_MIN)
    n_max = int(input(f"Maximum n_qubits [{DEFAULT_N_MAX}]: ") or DEFAULT_N_MAX)
    maxiter = int(input("LBFGSB max iterations (e.g. 50 or 300): "))

    print("\nChoose method:\n"
          " 1 = Global fidelity\n"
          " 2 = Local mean RDM\n"
          " 3 = Reverse 1q\n"
          " 4 = ALL\n")

    choice = input("Your choice: ").strip()

    if choice in ("2", "4"):
        k = int(input("k for RDM (1, 2, ...): ").strip())
    else:
        k = None

    if choice in ("1", "4"):
        benchmark_global_fidelity(folder_name, n_min, n_max, maxiter)

    if choice in ("2", "4") and k is not None:
        benchmark_local_mean(folder_name, n_min, n_max, maxiter, k)

    if choice in ("3", "4"):
        benchmark_reverse_1q(folder_name, n_min, n_max, maxiter)


if __name__ == "__main__":
    main()
