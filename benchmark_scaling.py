import os
import time
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

from configs.settings import default_benchmark_min, default_benchmark_max


DATA_INPUT_ROOT = "data_input"
DATA_OUTPUT_ROOT = "data_output"

# ansatz uses CRY so n >= 2
DEFAULT_N_MIN = default_benchmark_min
DEFAULT_N_MAX = default_benchmark_max


def get_or_create_vector(folder_name: str, n_qubits: int) -> np.ndarray:
    """
    Ensure there is a data vector of length 2**n_qubits in
    data_input/<folder_name>/n{n_qubits}_vec0.npy.

    If the file exists, load and return it.
    If not, create a new random vector, save, and return it.
    """
    folder_path = os.path.join(DATA_INPUT_ROOT, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    filename = f"n{n_qubits}_vec0.npy"
    path = os.path.join(folder_path, filename)

    if os.path.exists(path):
        vec = np.load(path)
        if vec.ndim != 1 or vec.shape[0] != 2 ** n_qubits:
            raise ValueError(
                f"Existing file {path} has shape {vec.shape}, "
                f"expected 1D of length {2**n_qubits}."
            )
        return vec

    # create new random vector: standard normal entries
    rng = np.random.default_rng(seed=make_seed(f"{folder_name}_data_n{n_qubits}"))
    vec = rng.standard_normal(2 ** n_qubits, dtype=float)

    np.save(path, vec)
    print(f"  Created new data vector for n={n_qubits} at {path}")
    return vec

import math

def auto_layers(n_qubits: int) -> int:
    """
    Minimal number of layers such that #parameters >= #amplitudes.
    Each layer has 2*n_qubits parameters.
    We require: layers * 2*n >= 2**n.
    """
    return math.ceil((2**n_qubits) / (2 * n_qubits))

def build_circuits_from_vector(ip: np.ndarray, n_qubits: int, num_layers: int):
    """
    Given a data vector ip of length 2**n_qubits, build the original and
    approximate circuits and return (qc, qc_app, target, training_params).
    """
    a, x = processData(ip)
    qc = build_sp_circuit(a, x)
    target = get_state_vector(qc)
    qc_app, training_params = build_app_circuit(n_qubits, int(num_layers))
    return qc, qc_app, target, training_params


def benchmark_global_fidelity(folder_name: str, n_min: int, n_max: int, maxiter: int):    
    results = []

    for n in range(n_min, n_max + 1):
        num_layers = auto_layers(n)
        num_params = 2 * n * num_layers
        print(f"\n[Global Fidelity] n = {n} qubits")
        print(f"  Using {num_layers} layers → {num_params} trainable parameters")
        ip = get_or_create_vector(folder_name, n)
        if ip.shape[0] != 2 ** n:
            raise ValueError(f"Vector length {ip.shape[0]} is not 2**{n}")

        qc, qc_app, target, training_params = build_circuits_from_vector(
            ip, n, num_layers
        )

        # set globals for training_global.losses
        L_global.training_params = training_params
        L_global.qc_app = qc_app
        L_global.target = target

        initial_params = generate_initial_params(
            len(training_params), seed=make_seed(f"benchmark_n{n}")
        )

        t0 = time.time()
        result_app = optimize_with_lbfgsb(
            loss_fidelity, initial_params, maxiter=int(maxiter)
        )
        t1 = time.time()

        runtime = t1 - t0
        final_loss = float(result_app.fun)
        print(f"  time = {runtime:.4f} s, final loss = {final_loss:.6e}")

        results.append((n, runtime, final_loss))

    # save to file
    os.makedirs(DATA_OUTPUT_ROOT, exist_ok=True)
    out_path = os.path.join(
        DATA_OUTPUT_ROOT,
        f"benchmark_global_fidelity_{folder_name}.txt",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Benchmark: Global Fidelity (LBFGSB)\n")
        f.write(f"dataset_folder: {folder_name}\n\n")
        f.write("{:>6s} {:>15s} {:>18s}\n".format("n", "time[s]", "final_loss"))
        f.write("-" * 45 + "\n")
        for n, t, loss in results:
            f.write("{:>6d} {:>15.6f} {:>18.6e}\n".format(n, t, loss))

    print(f"\nGlobal fidelity benchmark saved to: {out_path}")


def benchmark_local_mean(
    folder_name: str, n_min: int, n_max: int, maxiter: int, k: int
): 
    results = []

    for n in range(n_min, n_max + 1):
        num_layers = auto_layers(n)
        num_params = 2 * n * num_layers
        print(f"\n[Local Mean RDM, k={k}] n = {n} qubits")
        print(f"  Using {num_layers} layers → {num_params} trainable parameters")
        ip = get_or_create_vector(folder_name, n)
        if ip.shape[0] != 2 ** n:
            raise ValueError(f"Vector length {ip.shape[0]} is not 2**{n}")

        qc, qc_app, target, training_params = build_circuits_from_vector(
            ip, n, num_layers
        )

        # set globals for local/global losses
        L_local.training_params = training_params
        L_local.qc_app = qc_app
        L_local.target = target

        L_global.training_params = training_params
        L_global.qc_app = qc_app
        L_global.target = target

        import training_local.train_rdm_mean as T_mean

        T_mean.qc_app = qc_app

        loss_k_mean = make_kq_rdm_loss(k)

        initial_params = generate_initial_params(
            len(training_params), seed=make_seed(f"benchmark_n{n}")
        )

        t0 = time.time()
        result_app = train_rdm_lbfgsb(
            initial_params,
            loss_fn=loss_k_mean,
            maxiter=int(maxiter),
            label=f"{k}q mean RDM (benchmark n={n})",
        )
        t1 = time.time()

        runtime = t1 - t0
        global_fids = result_app.history["global_fidelity"]
        best_loss = float(np.min(global_fids))
        print(f"  time = {runtime:.4f} s, best global loss = {best_loss:.6e}")

        results.append((n, runtime, best_loss))

    os.makedirs(DATA_OUTPUT_ROOT, exist_ok=True)
    out_path = os.path.join(
        DATA_OUTPUT_ROOT,
        f"benchmark_local_mean_k{k}_{folder_name}.txt",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Benchmark: Local Mean RDM (k={k}, LBFGSB)\n")
        f.write(f"dataset_folder: {folder_name}\n\n")
        f.write("{:>6s} {:>15s} {:>18s}\n".format("n", "time[s]", "best_global_loss"))
        f.write("-" * 45 + "\n")
        for n, t, loss in results:
            f.write("{:>6d} {:>15.6f} {:>18.6e}\n".format(n, t, loss))

    print(f"\nLocal mean RDM benchmark saved to: {out_path}")


def benchmark_reverse_1q(folder_name: str, n_min: int, n_max: int, maxiter: int):    
    results = []

    for n in range(n_min, n_max + 1):
        num_layers = auto_layers(n)
        num_params = 2 * n * num_layers
        print(f"\n[Reverse 1q local loss] n = {n} qubits")
        print(f"  Using {num_layers} layers → {num_params} trainable parameters")
        ip = get_or_create_vector(folder_name, n)
        if ip.shape[0] != 2 ** n:
            raise ValueError(f"Vector length {ip.shape[0]} is not 2**{n}")

        qc, qc_app, target, training_params = build_circuits_from_vector(
            ip, n, num_layers
        )

        # set globals for global losses (used inside reverse loss)
        L_global.training_params = training_params
        L_global.qc_app = qc_app
        L_global.target = target

        # precompute for reverse method
        target_sv = Statevector.from_instruction(qc)
        target_amp = target_sv.data
        qc_app_inv = qc_app.inverse()

        L_rev.qc_app = qc_app
        L_rev.qc = qc
        L_rev.target_sv = target_sv
        L_rev.target_amp = target_amp
        L_rev.qc_app_inv = qc_app_inv

        initial_params = generate_initial_params(
            len(training_params), seed=make_seed(f"benchmark_n{n}")
        )

        t0 = time.time()
        result_app = optimize_with_lbfgsb(
            loss_1q_mean_reverse, initial_params, maxiter=int(maxiter)
        )
        t1 = time.time()

        runtime = t1 - t0
        final_loss = float(result_app.fun)
        print(f"  time = {runtime:.4f} s, final loss = {final_loss:.6e}")

        results.append((n, runtime, final_loss))

    os.makedirs(DATA_OUTPUT_ROOT, exist_ok=True)
    out_path = os.path.join(
        DATA_OUTPUT_ROOT,
        f"benchmark_reverse_1q_{folder_name}.txt",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Benchmark: Reverse 1q Local Loss (LBFGSB)\n")
        f.write(f"dataset_folder: {folder_name}\n\n")
        f.write("{:>6s} {:>15s} {:>18s}\n".format("n", "time[s]", "final_loss"))
        f.write("-" * 45 + "\n")
        for n, t, loss in results:
            f.write("{:>6d} {:>15.6f} {:>18.6e}\n".format(n, t, loss))

    print(f"\nReverse 1q benchmark saved to: {out_path}")


def main():
    print("=== Benchmark Scaling for Selected Methods (auto-generate data per n) ===\n")

    folder_name = input("Dataset folder name inside 'data_input' to use/create: ").strip()

    n_min = int(input(f"Minimum n_qubits to test [{DEFAULT_N_MIN}]: ") or DEFAULT_N_MIN)
    if n_min < 2:
        print("Ansatz uses CRY gates and needs at least 2 qubits – setting n_min = 2.")
        n_min = 2

    n_max = int(input(f"Maximum n_qubits to test [{DEFAULT_N_MAX}]: ") or DEFAULT_N_MAX)
    if n_max < n_min:
        raise ValueError("n_max must be >= n_min")

    maxiter = int(input("Number of LBFGSB iterations per n (e.g. 20): ").strip())

    print(
        "\nWhich method to benchmark?\n"
        "  1 = Global fidelity (LBFGSB)\n"
        "  2 = Local mean RDM (LBFGSB)\n"
        "  3 = Reverse 1q local loss (LBFGSB)\n"
        "  4 = ALL of the above\n"
    )
    choice = input("Your choice: ").strip()

    if choice in ("2", "4"):
        k = int(input("k for mean RDM (number of qubits in RDM, e.g. 1/2/3): ").strip())
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
