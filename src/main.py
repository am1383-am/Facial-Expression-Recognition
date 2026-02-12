import os
import pickle
from preprocessing.data_loader import get_data_generators
from models.baseline_model import build_model
from training.train import train_model
from utils.plot_results import plot_history
from evaluation.evaluator import evaluate_model
from keras.models import load_model

EMOTIONS = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'sad': 4, 'surprise': 5, 'neutral': 6, 'contempt': 7
}

def main():
    csv_path = 'data/raw/dataset_cleaned.csv'
    img_dir = 'data/raw/'
    model_path = 'models/baseline_model.keras'
    history_path = 'models/history.pkl'

    print("--- Phase 1: Data Preparation ---")
    train_gen, val_gen, test_gen = get_data_generators(csv_path, img_dir)
    
    print(f"Target Emotions: {list(EMOTIONS.keys())}")

    # print("\n" + "="*40)
    # print("Choose Mode:")
    # print("1. Load existing model")
    # print("2. Train new model")
    # choice = input("Enter 1 or 2: ").strip()
    # print("="*40 + "\n")
    choice = '2'
    model = None

    if choice == '1':
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            model = load_model(model_path)
            print("Model loaded successfully.")
        else:
            print(f"Error: Model file not found at {model_path}")
            print("Please train the model first (Select option 2).")
            return

    elif choice == '2':
        print("--- Phase 2: Model Building (New) ---")
        model = build_model(num_classes=len(EMOTIONS))
        model.summary()

        print("--- Phase 3: Training ---")
        history = train_model(model, train_gen, val_gen, epochs=30)

        print(f"Saving training history to {history_path}...")
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)

        print("Plotting training history...")
        plot_history(history)
        
        print(f"Saving model to {model_path}...")
        model.save(model_path)
        print("Model saved.")

    else:
        print("Invalid choice. Exiting.")
        return

    if model:
        print("--- Phase 4: Model Evaluation ---")
        evaluate_model(model, test_gen, EMOTIONS)
        print("Evaluation sequence finished.")

if __name__ == "__main__":
    main()