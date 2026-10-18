import numpy as np
from maths.Adam import Adam
from training_global.losses import make_sparse_amplitude_loss, loss_fidelity
from maths.numerical_grad import numerical_grad

qc_app = None

def random_subset_fidelity_training_adam(
    initial_params,
    num_cycles,
    block_size,
    batch_size_amplitudes,
    lr=0.01,
    delta=1e-5,
    seed=0
):
    rng = np.random.default_rng(seed)
    params = initial_params.copy()
    num_params = len(params)
    dim = 2 ** qc_app.num_qubits

    optimizer = Adam(lr=lr)
    history = []

    for cycle in range(num_cycles):

        # 1. Random amplitude subset S
        # allow static int, list schedule, or callable schedule
        if callable(batch_size_amplitudes):
            batch_size = batch_size_amplitudes(cycle)  # function schedule
        elif isinstance(batch_size_amplitudes, (list, tuple, np.ndarray)):
            batch_size = batch_size_amplitudes[cycle]  # list schedule
        else:
            batch_size = batch_size_amplitudes         # static integer

        batch = min(batch_size, dim)

        amplitude_indices = rng.choice(dim, size=batch, replace=False)
        sparse_loss = make_sparse_amplitude_loss(amplitude_indices) # insert the randomly chosen indices
        # this sparse_loss now fixed for only a set of random indices chosen from above
        
        # 2. Random subset of parameters B
        if block_size >= num_params:
            block = np.arange(num_params)
        else:
            block = rng.choice(num_params, size=block_size, replace=False)
        block = np.sort(block)

        # 3. Restrict the loss to just params in B (relevant if train with a random subset of thetas)
        def block_loss(block_params):
            full = params.copy()
            full[block] = block_params  # update the chosen active params to get the loss 
            return sparse_loss(full)

        # 4. Compute gradient on this block
        block_params = params[block].copy()
        block_grads = numerical_grad(block_loss, block_params, delta=delta)

        # 5. Adam update only on that block
        updated_block = optimizer.step(block_params, block_grads) # run ADAM once on the block of thetas
        params[block] = updated_block # update the current params

        # 6. Logging
        s_loss = sparse_loss(params)
        try:
            g_loss = loss_fidelity(params)
        except:
            g_loss = None

        print(
            f"Cycle {cycle+1:3d}/{num_cycles} | "
            f"S={batch:2d} | "
            f"sparse_loss={s_loss:.3e} | "
            f"global_loss={g_loss:.3e}"
        )

        history.append({
            "params": params.copy(),
            "block_indices": block.copy(),
            "amplitudes": amplitude_indices.copy(),
            "sparse_loss": s_loss,
            "global_loss": g_loss
        })

    return params, history
