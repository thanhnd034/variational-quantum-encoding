from scipy.optimize import minimize
from configs.settings import default_maxiter, default_print_modulo
from training_global.losses import loss_fidelity
from maths.numerical_grad import numerical_grad


def optimize_with_lbfgsb(
    loss_fn,
    initial_params,
    global_fidelity_fn=loss_fidelity,    # this returns a LOSS = 1 - fidelity
    maxiter=default_maxiter,
    global_threshold=1e-10               # stop when global loss <= threshold
):

    # ---- history tracking dict ----
    history = {
        "params": [],
        "loss": [],
        "global_fidelity": []        # keep original name for compatibility
    }

    iteration = {"count": 0}

    # ---- callback triggered once per iteration ----
    def callback(params):
        iteration["count"] += 1

        # evaluate losses
        local_loss = loss_fn(params)
        global_loss = global_fidelity_fn(params)       # this *is* a loss

        # record history
        history["params"].append(params.copy())
        history["loss"].append(local_loss)
        history["global_fidelity"].append(global_loss)

        # print global fidelity loss
        if iteration["count"] % int(default_print_modulo) == 0:
            print(f"Iteration {iteration['count']}: Global fidelity loss = {global_loss:.10f}")

        # ---- custom stop condition based on global fidelity *loss* ----
        if global_loss <= global_threshold:
            print(f"Reached global fidelity loss threshold ({global_loss:.3e} <= {global_threshold:.3e})")
            raise StopIteration


    # ---- run L-BFGS-B ----
    try:
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

        # if minimize finishes normally, attach history and return
        result.history = history
        return result

    except StopIteration:
        # ---- construct a dummy result object with .history ----
        print("Manual StopIteration triggered — returning history only.")

        class DummyResult:
            pass

        final = DummyResult()
        final.x = history["params"][-1]
        final.fun = history["loss"][-1]
        final.history = history
        return final
