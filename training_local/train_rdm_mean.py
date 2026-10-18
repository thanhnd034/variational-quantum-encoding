from itertools import combinations
from training_local.losses import local_rdm_loss
import numpy as np
from training_global.losses import loss_fidelity
from maths.optimize_with_lbfgsb import optimize_with_lbfgsb

qc_app = None

def make_kq_rdm_loss(k):
    """
    Generic RDM loss builder.
    Returns a function(params) that computes the mean k-qubit RDM loss.
    
    Example:
        loss_fn = make_kq_rdm_loss(2)
        loss_fn(params)   -> avg 2q RDM loss
    """
    def loss_fn(params):
        N = qc_app.num_qubits
        subsets = combinations(range(N), k)
        
        losses = []
        for subset in subsets:
            subset = list(subset)        # convert tuple -> list
            losses.append(local_rdm_loss(params, subset=subset))

        return np.mean(losses)
    
    return loss_fn


def train_rdm_lbfgsb(
    initial_params,
    loss_fn,
    maxiter=400,
    label="RDM"
):
    """
    Generic LBFGSB trainer for ANY RDM loss function.

    Args:
        initial_params  : vector of θ
        loss_fn         : function(params) -> scalar loss
        maxiter         : LBFGSB iterations
        label           : name for printing ("1q", "2q", "3q"...)
    """

    result = optimize_with_lbfgsb(
        loss_fn=loss_fn,
        initial_params=initial_params,
        global_fidelity_fn=loss_fidelity,
        maxiter=maxiter
    )

    final_params = result.x

    # compute final RDM loss
    final_loss = loss_fn(final_params)

    # compute global fidelity if available
    try:
        global_loss = loss_fidelity(final_params)
    except Exception:
        global_loss = None

    print(f"\n=== {label} LBFGSB Training Finished ===")
    print(f"Final {label} loss: {final_loss:.6e}")
    print(f"Global fidelity loss: {global_loss:.6e}")

    return result