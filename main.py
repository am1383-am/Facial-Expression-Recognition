import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.data_preparation.data_preparation import run_preparation
except ImportError as e:
    def run_preparation(): print("Error: Data Preparation module not found in src/data_preparation/.")

try:
    from src.training_pipeline import main as run_training_pipeline
except ImportError as e:
    def run_training_pipeline(): print("Error: Training module not found in src/.")

try:
    from app import run_from_main
except ImportError as e:
    print(f"Warning: app.py not found in root: {e}")
    def run_from_main(): print("Error: app.py not found. Please create it first.")

def show_menu():
    print("\n" + "="*50)
    print("     Facial Expression Recognition Project")
    print("="*50)
    print("Please select an operation mode:")
    print("1. [Data Prep]  Download & Prepare Dataset")
    print("2. [Training]   Train Model & Evaluate")
    print("3. [Demo]       Run Live Demo (Gradio)")
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
            print("\n>>> Launching Gradio Demo...")
            run_from_main()

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