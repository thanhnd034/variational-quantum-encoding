import os
import numpy as np
import json

def read_and_print_dataset(folder_name):
    base_dir = "data_input"
    folder_path = os.path.join(base_dir, folder_name)

    # Check folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_name}' not found inside data_input/")
        return

    print(f"=== Dataset: {folder_name} ===\n")

    # Try reading metadata
    metadata_path = os.path.join(folder_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print("Metadata:", metadata)
    else:
        print("No metadata.json found.")

    # List .npy files
    npy_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".npy"))
    print(f"Number of data vectors: {len(npy_files)}\n")

    # Print each vector
    for f_name in npy_files:
        vec = np.load(os.path.join(folder_path, f_name))
        print(f"→ {f_name}")
        print(vec)
        print()

    print("=== End of dataset ===\n")
