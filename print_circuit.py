from utilities.load_dataset import load_dataset
from circuits.helpers import processData
from circuits.original import build_sp_circuit

print("=== Circuit Builder ===")

while True:
    folder_name = input("Enter the name of the dataset folder inside 'data_input': ").strip()
    data_number = input("Creating the circuit for data vector number: ").strip()

    # Call the helper
    dataset = load_dataset(folder_name)
    ip = dataset[int(data_number)]
    print(ip)
    
    a,x = processData(ip)
    qc = build_sp_circuit(a,x)
    print(qc.draw(output='text'))
    

    cont = input("\nCreate circuit for another data vector? (y/n): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("\nExiting circuit builder.")
        break

    print()