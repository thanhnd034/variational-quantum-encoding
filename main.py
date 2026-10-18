import subprocess
import sys

def run_script(path):
    try:
        subprocess.run([sys.executable, path], check=True)
    except Exception as e:
        print(f"❌ Error running {path}: {e}")

def main_menu():
    print("=== Quantum Encoding Project Menu ===")

    while True:
        print("\nChoose an option:")
        print(" 1. Data Generator")
        print(" 2. Print Circuit")
        print(" 3. Print Approx Circuit")
        print(" 4. Train Global Fidelity")
        print(" 5. Train Global Fidelity (with schedule)")
        print(" 6. Train Local RDM Loss with N random Qubits (using Adam)")
        print(" 7. Train Mean Local RDM Loss with N Qubits (using lbfgsb)")
        print(" 8. Train Reverse Local RDM Loss with 1 Qubit (using lbfgsb)")
        print(" 9. Train Multiple Data Vectors using Reverse Method (option 8)")
        print(" 10. Benchmarking Methods")
        print(" 11. Benchmarking Methods with fixed threshold")
        print(" 0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            run_script("data_generator.py")

        elif choice == "2":
            run_script("print_circuit.py")

        elif choice == "3":
            run_script("print_approx.py")

        elif choice == "4":
            run_script("train_global_fidelity.py")

        elif choice == "5":
            run_script("train_global_fidelity_with_schedule.py")

        elif choice == "6":
            run_script("train_local_random_rdm.py")

        elif choice == "7":
            run_script("train_local_mean_rdm.py") 

        elif choice == "8":
            run_script("train_reverse_local_loss.py")        

        elif choice == "9":
            run_script("train_multiple_reverse.py")

        elif choice == "10":
            run_script("benchmark_scaling.py") 

        elif choice == "11":
            run_script("benchmark_threshold.py")                      

        elif choice == "0":
            print("\nExiting. Goodbye!")
            break

        else:
            print("❌ Invalid choice. Try again.")


if __name__ == "__main__":
    main_menu()
