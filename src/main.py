import os
from preprocessing.data_loader import get_data_generators
from models.final_model import build_final_model 
from training.train import train_model
from evaluation.evaluate import evaluate_model 

EMOTIONS = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6, 'contempt': 7}

def main():
    if not os.path.exists('results'):
        os.makedirs('results')

    print("\n--- Phase 2: Data Loading ---")
    train_gen, val_gen, test_gen = get_data_generators('data/raw/dataset_cleaned.csv', 'data/raw/', batch_size=32)

    print("\n--- Phase 2: Building Final Model ---")
    model = build_final_model(num_classes=len(EMOTIONS))
    model.summary()

    print("\n--- Phase 2: Training ---")
    history = train_model(model, train_gen, val_gen, epochs=40)

    print("\n--- Phase 2: Detailed Evaluation ---")
    evaluate_model(model, test_gen, EMOTIONS)
    
    print("\n Phase 2 Cycle Completed.")

if __name__ == "__main__":
    main()