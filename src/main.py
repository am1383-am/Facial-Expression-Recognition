from preprocessing.data_loader import get_data_generators
from models.baseline_model import build_model
from training.train import train_model

EMOTIONS = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'sad': 4, 'surprise': 5, 'neutral': 6, 'contempt': 7
}

def main():
    print("--- Phase 1: Data Preparation ---")
    train_gen, val_gen, test_gen = get_data_generators('data/raw/dataset_cleaned.csv', 'data/raw/')
    
    print(f"Target Emotions: {list(EMOTIONS.keys())}")

    print("--- Phase 2: Model Building ---")
    model = build_model(num_classes=len(EMOTIONS))
    model.summary()

    print("--- Phase 3: Training Baseline ---")
    train_model(model, train_gen, val_gen, epochs=30)
    
    print("Training sequence finished.")

if __name__ == "__main__":
    main()