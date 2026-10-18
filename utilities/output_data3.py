import os
import numpy as np
import json

def output_data3(input_folder_name, data_vector_number, final_params, history):
    """
    Saves the output of random_subset_fidelity_training_adam into:
        data_output/<input_folder_name>/vector_<data_vector_number>/

    Exports both .npy and .txt versions for all arrays.
    """

    # Base output folder
    base_dir = "data_output"
    os.makedirs(base_dir, exist_ok=True)

    # Folder for this dataset
    dataset_dir = os.path.join(base_dir, input_folder_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Folder for this specific data vector
    vector_dir = os.path.join(dataset_dir, f"vector_{data_vector_number}")
    os.makedirs(vector_dir, exist_ok=True)

    # -----------------------------------------
    # Helper: save .npy and .txt
    # -----------------------------------------
    def save_both(arr, name):
        npy_path = os.path.join(vector_dir, f"{name}.npy")
        txt_path = os.path.join(vector_dir, f"{name}.txt")

        # save npy
        np.save(npy_path, arr)

        # save txt
        try:
            np.savetxt(txt_path, arr, fmt="%.18f")
        except:
            # ragged arrays or object arrays
            with open(txt_path, "w") as f:
                for item in arr:
                    f.write(str(item) + "\n")

    # -----------------------------------------
    # Save final params
    # -----------------------------------------
    save_both(np.array(final_params), "final_params")

    # -----------------------------------------
    # Build history arrays
    # -----------------------------------------
    local_losses = np.array([h["local_loss"] for h in history])
    global_losses = np.array([h["global_loss"] for h in history])
    subsets = np.array([h["subset"] for h in history], dtype=object)
    blocks = np.array([h["block"] for h in history], dtype=object)
    params_over_time = np.array([h["params"] for h in history], dtype=object)

    # -----------------------------------------
    # Save history
    # -----------------------------------------
    save_both(local_losses, "history_local_loss")
    save_both(global_losses, "history_global_loss")
    save_both(blocks, "history_blocks")
    save_both(subsets, "history_subsets")
    save_both(params_over_time, "history_params")

    # -----------------------------------------
    # Metadata
    # -----------------------------------------
    metadata = {
        "num_cycles": len(history),
        "final_local_loss": float(local_losses[-1]),
        "final_global_loss": float(global_losses[-1]) if global_losses[-1] is not None else None,
        "num_params": len(final_params)
    }

    with open(os.path.join(vector_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n✅ Saved ADAM training results (npy + txt) to: {vector_dir}\n")
