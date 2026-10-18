from utilities.data_generation import generate_random_data_vectors

print("=== Random Data Vector Generator ===")

while True:
    try:
        n_qubits = int(input("Number of qubits: "))
        num_vectors = int(input("Number of data vectors: "))
        folder_name = input("Folder name: ")


        generate_random_data_vectors(
            n_qubits=n_qubits,
            num_vectors=num_vectors,
            folder_name=folder_name
        )

    except ValueError:
        print("\n❌ Invalid input. Please enter correct numbers.\n")
        continue

    cont = input("Generate another dataset? (y/n): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("\nExiting. Goodbye!")
        break

    print()