import sys
import os
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.data_preparation.data_preparation import run_preparation
except ImportError as e:
    def run_preparation(): print("Error: Data Preparation module not found in src/data_preparation/.")

try:
    from src.training_pipeline import main as run_training_pipeline
except ImportError as e:
    def run_training_pipeline(): print("Error: Training module not found in src/.")

def run_streamlit_demo():
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: '{app_path}' not found. Please create app.py first.")
        return

    print("\n>>> Launching Streamlit UI...")
    print(">>> Press Ctrl+C in the terminal to stop the demo and return to menu.\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    except KeyboardInterrupt:
        print("\n>>> Streamlit stopped by user.")
    except Exception as e:
        print(f"\n>>> Error launching Streamlit: {e}")

def show_menu():
    print("\n" + "="*50)
    print("     Facial Expression Recognition Project")
    print("="*50)
    print("Please select an operation mode:")
    print("1. [Data Prep]  Download & Prepare Dataset")
    print("2. [Training]   Train Model & Evaluate")
    print("3. [Demo]       Run Live Demo (Streamlit)")
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
            run_streamlit_demo()

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