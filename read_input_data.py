from utilities.read_print_dataset import read_and_print_dataset

print("=== Data Reader ===")

while True:
    folder_name = input("Enter the name of the dataset folder inside 'data_input': ").strip()

    # Call the helper
    read_and_print_dataset(folder_name)

    cont = input("\nRead another dataset? (y/n): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("\nExiting Data Reader.")
        break

    print()