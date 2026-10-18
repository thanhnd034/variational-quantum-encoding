import os
import numpy as np

def print_stats(arr):
    """Print useful statistical information about a numpy array."""
    print("Shape:", arr.shape)
    print("Dtype:", arr.dtype)

    if arr.dtype.kind in "iu":         # int or unsigned
        print("Min:", arr.min(), "Max:", arr.max())
    elif arr.dtype.kind == "f":        # float
        print("Min:", arr.min(), "Max:", arr.max(), "Mean:", arr.mean())
    else:
        print("(Non-numeric or object array)")
    np.set_printoptions(threshold=np.inf)
    print("\nData preview:")
    print(arr)


def npy_browser_data_output():
    base_dir = "data_output"

    if not os.path.isdir(base_dir):
        print("❌ data_output folder not found!")
        return

    print("=== .npy Browser (data_output only) ===")

    while True:
        # --- Step 1: pick dataset folder ---
        datasets = sorted([d for d in os.listdir(base_dir)
                           if os.path.isdir(os.path.join(base_dir, d))])

        if not datasets:
            print("❌ No dataset folders inside data_output/")
            return

        print("\nAvailable datasets:")
        for i, d in enumerate(datasets):
            print(f" [{i}] {d}")

        try:
            ds_idx = int(input("\nSelect dataset number: "))
            dataset_name = datasets[ds_idx]
        except Exception:
            print("Invalid selection.")
            continue

        dataset_path = os.path.join(base_dir, dataset_name)

        # --- Step 2: pick vector folder ---
        vectors = sorted([v for v in os.listdir(dataset_path)
                          if os.path.isdir(os.path.join(dataset_path, v))])

        if not vectors:
            print("❌ No vector folders in this dataset.")
            continue

        print(f"\nVectors in {dataset_name}:")
        for i, v in enumerate(vectors):
            print(f" [{i}] {v}")

        try:
            v_idx = int(input("\nSelect vector folder number: "))
            vector_folder = vectors[v_idx]
        except Exception:
            print("Invalid selection.")
            continue

        vector_path = os.path.join(dataset_path, vector_folder)

        # --- Step 3: pick .npy file ---
        npy_files = sorted([f for f in os.listdir(vector_path) if f.endswith(".npy")])

        if not npy_files:
            print("❌ No .npy files in this vector folder.")
            continue

        print(f"\nFound .npy files in {vector_folder}:")
        for i, f in enumerate(npy_files):
            print(f" [{i}] {f}")

        try:
            f_idx = int(input("\nSelect file number: "))
            fname = npy_files[f_idx]
        except Exception:
            print("Invalid selection.")
            continue

        file_path = os.path.join(vector_path, fname)
        print(f"\n=== Loading {file_path} ===\n")

        arr = np.load(file_path, allow_pickle=True)
        print_stats(arr)

        cont = input("\nBrowse another file? (y/n): ").lower()
        if cont not in ["y", "yes"]:
            break

    print("\nExiting browser. Goodbye!")


if __name__ == "__main__":
    npy_browser_data_output()