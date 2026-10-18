import numpy as np
from configs.settings import initial_min_value_params, initial_max_value_params

def generate_initial_params(num_params, low=initial_min_value_params, high=initial_max_value_params, seed=None):
    rng = np.random.default_rng(seed)   
    return rng.uniform(low, high, num_params)