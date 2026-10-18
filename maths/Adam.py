import numpy as np
from configs.settings import default_lr, default_beta1, default_beta2, default_eps

class Adam:
    def __init__(self, lr=default_lr, beta1=default_beta1, beta2=default_beta2, eps=default_eps):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0 # iteration counter

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads # momentum and RMS
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads * grads)

        m_hat = self.m / (1 - self.beta1 ** self.t) # bias correction
        v_hat = self.v / (1 - self.beta2 ** self.t)

        updated = params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return updated