from utilities.load_dataset import load_dataset
from circuits.helpers import processData
from circuits.approx import build_app_circuit
import numpy as np

print("=== Variational Circuit Builder ===")

while True:
    folder_name = input("Enter the name of the dataset folder inside 'data_input': ").strip()
    data_number = input("Creating the circuit for data vector number: ").strip()
    num_layers = input("Insert the number of layers: ").strip()

    # Call the helper
    dataset = load_dataset(folder_name)
    ip = dataset[int(data_number)]
    print(ip)
    
    qc_app, training_params = build_app_circuit(int(np.log2(len(ip))), int(num_layers))
    print(f"\nNumber of parameters to train: {len(training_params)}")
    print(qc_app.draw(output='text'))
    

    cont = input("\nCreate circuit for another data vector? (y/n): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("\nExiting circuit builder.")
        break

    print()