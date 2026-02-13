import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.data_preparation.data_preparation import run_preparation
except ImportError as e:
    print(f"Error importing Data Preparation module: {e}")
    def run_preparation(): print("Error: Data Preparation module not found.")

try:
    from src.training_pipeline import main as run_training_pipeline
except ImportError as e:
    print(f"Error importing Training module: {e}")
    def run_training_pipeline(): print("Error: Training module not found.")

def run_demo():
    print("\n" + "*"*50)
    print("* DEMO MODE (Coming Soon)             *")
    print("*"*50)
    print("This feature is currently under construction.")
    print("Please implement the demo logic in src/demo/...")
    input("\nPress Enter to return to menu...")

def show_menu():
    print("\n" + "="*50)
    print("     Facial Expression Recognition Project")
    print("="*50)
    print("Please select an operation mode:")
    print("1. [Data Prep]  Download & Prepare Dataset")
    print("2. [Training]   Train Model & Evaluate")
    print("3. [Demo]       Run Live Demo")
    print("0. Exit")
    print("-" * 50)

def main():
    while True:
        show_menu()
        choice = input("Enter your choice (0-3): ").strip()

        if choice == '1':
            print("\n>>> Starting Data Preparation...")
            run_preparation()
            print("\n>>> Data Preparation Finished.")
            
        elif choice == '2':
            print("\n>>> Starting Training Pipeline...")
            run_training_pipeline()
            print("\n>>> Training Pipeline Finished.")

        elif choice == '3':
            run_demo()

        elif choice == '0':
            print("\nExiting program. Goodbye!")
            break

        else:
            print("\n[!] Invalid option. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
        sys.exit(0)