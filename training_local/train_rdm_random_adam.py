from itertools import combinations
from maths.numerical_grad import numerical_grad
from maths.Adam import Adam
import numpy as np
from training_local.losses import local_rdm_loss
from training_global.losses import loss_fidelity

qc_app = None

def cycle_pairs(n):
    """Infinite generator cycling through all 2-qubit pairs."""
    pair_list = list(combinations(range(n), 2))
    idx = 0
    while True:
        yield list(pair_list[idx])
        idx = (idx + 1) % len(pair_list)


class RDMTrainerAdam:
    def __init__(
        self, 
        qc_app, 
        local_rdm_loss, 
        loss_fidelity,
        default_lr=0.01,
        default_delta=1e-5,
    ):
        """
        qc_app: QuantumCircuit for simulation
        local_rdm_loss: function(params, subset) -> local RDM loss
        loss_fidelity: function(params) -> global fidelity loss
        default_lr: fallback learning rate
        default_delta: fallback finite-difference step
        """
        self.qc_app = qc_app
        self.local_rdm_loss = local_rdm_loss
        self.loss_fidelity = loss_fidelity
        self.default_lr = default_lr
        self.default_delta = default_delta

    # ---------------------------------------------------------
    # Utility: finite-difference gradient
    # ---------------------------------------------------------
    def grad(self, fn, x, delta):
        return numerical_grad(fn, x, delta=delta)

    # ---------------------------------------------------------
    # Utility: one Adam update step
    # ---------------------------------------------------------
    def adam_step(self, optimizer, params, block, block_loss_fn, delta):
        block_params = params[block].copy()
        block_grads = self.grad(block_loss_fn, block_params, delta=delta)
        updated = optimizer.step(block_params, block_grads)
        params[block] = updated
        return params

    # ---------------------------------------------------------
    # Generic training loop (used by all methods)
    # ---------------------------------------------------------
    def _train(
        self,
        initial_params,
        num_cycles,
        block_size,
        subset_generator,
        rng,
        lr,
        delta,
    ):
        params = initial_params.copy()
        num_params = len(params)
        optimizer = Adam(lr=lr)
        history = []

        for cycle in range(num_cycles):

            # --- 1. Choose subset (1q, 2q, 3q, or RR) ---
            subset = subset_generator(rng)

            sparse_loss = lambda p: self.local_rdm_loss(p, subset=subset)

            # --- 2. Random parameter-block selection ---
            if block_size >= num_params:
                block = np.arange(num_params)
            else:
                block = np.sort(rng.choice(num_params, block_size, replace=False))

            # --- 3. Block-restricted loss ---
            def block_loss(block_params):
                full = params.copy()
                full[block] = block_params
                return sparse_loss(full)

            # --- 4. Adam update ---
            params = self.adam_step(optimizer, params, block, block_loss, delta)

            # --- 5. Compute losses ---
            local_loss_value = sparse_loss(params)

            try:
                global_loss_value = self.loss_fidelity(params)
            except Exception:
                global_loss_value = None

            # --- 6. Print progress ---
            print(
                f"Cycle {cycle+1:4d}/{num_cycles} | "
                f"subset={subset} | "
                f"local_loss={local_loss_value:.3e} | "
                f"global_loss={global_loss_value:.3e}"
            )

            # --- 7. Store history ---
            history.append({
                "params": params.copy(),
                "subset": subset,
                "local_loss": local_loss_value,
                "global_loss": global_loss_value,
                "block": block.copy(),
            })

        return params, history

    # =========================================================
    # PUBLIC TRAINING MODES
    # =========================================================

    # ---------------------------------------------------------
    # 1-qubit local RDM training
    # ---------------------------------------------------------
    def train_1q(
        self,
        initial_params,
        num_cycles,
        block_size,
        seed=0,
        lr=None,
        delta=None,
    ):
        rng = np.random.default_rng(seed)
        lr = lr if lr is not None else self.default_lr
        delta = delta if delta is not None else self.default_delta

        return self._train(
            initial_params,
            num_cycles,
            block_size,
            subset_generator=lambda rng: [int(rng.integers(0, self.qc_app.num_qubits))],
            rng=rng,
            lr=lr,
            delta=delta,
        )

    # ---------------------------------------------------------
    # 2-qubit random local RDM training
    # ---------------------------------------------------------
    def train_2q(
        self,
        initial_params,
        num_cycles,
        block_size,
        seed=0,
        lr=None,
        delta=None,
    ):
        rng = np.random.default_rng(seed)
        lr = lr if lr is not None else self.default_lr
        delta = delta if delta is not None else self.default_delta

        return self._train(
            initial_params,
            num_cycles,
            block_size,
            subset_generator=lambda rng: rng.choice(
                self.qc_app.num_qubits, size=2, replace=False
            ).tolist(),
            rng=rng,
            lr=lr,
            delta=delta,
        )

    # ---------------------------------------------------------
    # 3-qubit random local RDM training
    # ---------------------------------------------------------
    def train_3q(
        self,
        initial_params,
        num_cycles,
        block_size,
        seed=0,
        lr=None,
        delta=None,
    ):
        if self.qc_app.num_qubits < 3:
            raise ValueError("Need at least 3 qubits in qc_app.")

        rng = np.random.default_rng(seed)
        lr = lr if lr is not None else self.default_lr
        delta = delta if delta is not None else self.default_delta

        return self._train(
            initial_params,
            num_cycles,
            block_size,
            subset_generator=lambda rng: np.sort(
                rng.choice(self.qc_app.num_qubits, size=3, replace=False)
            ).tolist(),
            rng=rng,
            lr=lr,
            delta=delta,
        )

    # ---------------------------------------------------------
    # n-qubit random local RDM training
    # ---------------------------------------------------------
    def train_nq(
        self,
        num_qubit,
        initial_params,
        num_cycles,
        block_size,
        seed=0,
        lr=None,
        delta=None,
    ):
        if self.qc_app.num_qubits < num_qubit:
            raise ValueError(f"Need at least {num_qubit} qubits in qc_app.")

        rng = np.random.default_rng(seed)
        lr = lr if lr is not None else self.default_lr
        delta = delta if delta is not None else self.default_delta

        return self._train(
            initial_params,
            num_cycles,
            block_size,
            subset_generator=lambda rng: np.sort(
                rng.choice(self.qc_app.num_qubits, size=num_qubit, replace=False)
            ).tolist(),
            rng=rng,
            lr=lr,
            delta=delta,
        )