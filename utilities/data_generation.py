import os
import numpy as np
import json
from configs.settings import default_min_values_gen, default_max_values_gen

def generate_random_data_vectors(n_qubits, num_vectors, folder_name, min_val=default_min_values_gen, max_val=default_max_values_gen):
    """
    Generate random data vectors (each of length 2^n_qubits) and store them in:
        data_input/<folder_name>/
    
    Saves for each vector:
        vector_000.npy
        vector_000.txt
        vector_001.npy
        vector_001.txt
        ...
    
    Also saves:
        metadata.json
    """

    # Base folder
    base_dir = "data_input"
    os.makedirs(base_dir, exist_ok=True)

    # Full path to the new dataset folder
    target_dir = os.path.join(base_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    # Dimension of each data vector
    dim = 2 ** n_qubits

    # Generate & save vectors
    for i in range(num_vectors):
        vec = np.random.uniform(low=min_val, high=max_val, size=dim)

        # ---- Save .npy file ----
        npy_path = os.path.join(target_dir, f"vector_{i:03d}.npy")
        np.save(npy_path, vec)

        # ---- Save .txt file ----
        txt_path = os.path.join(target_dir, f"vector_{i:03d}.txt")
        # Format floats nicely
        np.savetxt(txt_path, vec, fmt="%.18f")

    # Save metadata
    metadata = {
        "n_qubits": n_qubits,
        "num_vectors": num_vectors,
        "min_val": min_val,
        "max_val": max_val,
        "vector_dimension": dim,
    }

    with open(os.path.join(target_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Generated {num_vectors} vectors in '{target_dir}' (with .npy and .txt files)")
