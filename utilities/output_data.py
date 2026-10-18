import os
import numpy as np
import json

def output_data(input_folder_name, data_vector_number, result):
    """
    Saves optimization results into:
        data_output/<input_folder_name>/vector_<data_vector_number>/
    and ALSO exports .txt versions of all arrays.
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

    history = result.history  # shorthand

    # =====================================================
    # Helper: save both .npy and .txt versions
    # =====================================================
    def save_both(array, name):
        npy_path = os.path.join(vector_dir, f"{name}.npy")
        txt_path = os.path.join(vector_dir, f"{name}.txt")

        # Save .npy
        np.save(npy_path, array)

        # Save .txt (handle object arrays separately)
        try:
            np.savetxt(txt_path, array, fmt="%.18f")
        except:
            # For object / ragged arrays → print each item on a new line
            with open(txt_path, "w") as f:
                for item in array:
                    f.write(str(item) + "\n")

    # =====================================================
    # Save optimal parameters
    # =====================================================
    save_both(result.x, "optimal_params")

    # =====================================================
    # Save final loss as both .txt and in metadata
    # =====================================================
    final_loss_path = os.path.join(vector_dir, "final_loss.txt")
    with open(final_loss_path, "w") as f:
        f.write(str(result.fun))

    # =====================================================
    # Save history arrays
    # =====================================================
    save_both(np.array(history["params"], dtype=object), "history_params")
    save_both(np.array(history["loss"]), "history_loss")
    save_both(np.array(history["global_fidelity"]), "history_global_fidelity")

    # =====================================================
    # Save metadata.json
    # =====================================================
    metadata = {
        "data_input_folder": input_folder_name,
        "data_vector_number": data_vector_number,
        "num_iterations": len(history["loss"]),
        "num_params": len(result.x)
    }
    with open(os.path.join(vector_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n✅ Saved training results (npy + txt) to: {vector_dir}\n")
