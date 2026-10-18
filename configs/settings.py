import numpy as np

# Master seed
MASTER_SEED = 2025

# Default number of variational layers
default_number_of_layers = 3

# Default values for data generation
default_max_values_gen = 100
default_min_values_gen = -100

# Initial parameters generation
initial_min_value_params = 0
initial_max_value_params = 2*np.pi


# Default values for numerical gradient
default_numerical_grad_delta = 1e-5


# Default lbfgsb values
default_maxiter = 500
default_print_modulo = 1

# Default Adam values
default_lr = 0.01
default_beta1 = 0.9
default_beta2 = 0.999
default_eps = 1e-5

# Training global fidelity with schedule (check the schedule file)
default_lr_schedule = 0.05
default_delta_schedule = 1e-3
default_seed_schedule = "Global Loss Sparse Amplitude MSE with Schedule"

# Training random RDM with Adam
default_lr_adam_random_rdm = 0.05
default_delta_adam_random_rdm = 1e-3

# Settings for benchmark
default_benchmark_min = 2 
default_benchmark_max = 5
TARGET_THRESHOLD = 1e-3