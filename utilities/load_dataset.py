import numpy as np
import os

def load_dataset(folder_name):
    folder_path = os.path.join("data_input", folder_name)
    
    # list all .npy files sorted
    files = sorted(f for f in os.listdir(folder_path) if f.endswith(".npy"))
    
    # load each vector into a list
    vectors = []
    for f in files:
        vec = np.load(os.path.join(folder_path, f))
        vectors.append(vec)

    data_set = np.array(vectors)
    return data_set


