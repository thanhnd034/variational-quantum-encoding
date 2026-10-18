import numpy as np
from configs.settings import default_numerical_grad_delta


def numerical_grad(loss_fn, params, delta=default_numerical_grad_delta):
    grads = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy();  p_plus[i] += delta
        p_minus = params.copy(); p_minus[i] -= delta
        grads[i] = (loss_fn(p_plus) - loss_fn(p_minus)) / (2 * delta)
    return grads
'''

def numerical_grad(loss_fn, params, delta=default_numerical_grad_delta):
    grads = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy();  p_plus[i] += np.pi/2
        p_minus = params.copy(); p_minus[i] -= np.pi/2
        grads[i] = (loss_fn(p_plus) - loss_fn(p_minus)) / (2.0)
    return grads

'''