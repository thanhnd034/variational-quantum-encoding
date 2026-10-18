from scipy.optimize import minimize
from configs.settings import default_maxiter, default_print_modulo
from training_global.losses import loss_fidelity
from maths.numerical_grad import numerical_grad

def optimize_with_lbfgsb(loss_fn, initial_params, global_fidelity_fn=loss_fidelity, maxiter=default_maxiter):

    # ---- history tracking dict ----
    history = {
        "params": [],           # stores xk per iteration
        "loss": [],             # stores loss_fn(xk)
        "global_fidelity": []   # stores global_fidelity_fn(xk)
    }

    iteration = {"count": 0}

    # ---- callback triggered once per iteration ----
    def callback(params):
        iteration["count"] += 1

        # record history
        history["params"].append(params.copy())
        history["loss"].append(loss_fn(params))
        history["global_fidelity"].append(global_fidelity_fn(params))

        # print global fidelity
        fidelity = history["global_fidelity"][-1]
        if iteration["count"] % int(default_print_modulo) == 0:
            print(f"Iteration {iteration['count']}: Global fidelity = {fidelity:.10f}")

    # ---- run L-BFGS-B ----
    result = minimize(
        fun=loss_fn,
        x0=initial_params,
        jac=lambda p: numerical_grad(loss_fn, p, delta=1e-5),
        method="L-BFGS-B",
        callback=callback,
        options={
            "maxiter": maxiter,
            "disp": True
        }
    )

    # attach history to the result object
    result.history = history

    return result


