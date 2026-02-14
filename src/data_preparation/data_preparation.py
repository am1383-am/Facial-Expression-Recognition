import kagglehub
import shutil
import os
from pathlib import Path
import csv
from tqdm import tqdm
import random
import time

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent 
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
CSV_FILE = DATA_RAW_DIR / "dataset.csv"
CLEANED_CSV_FILE = DATA_RAW_DIR / "dataset_cleaned.csv"

EMOTION_MAPPING = {
    'angry': 0, 'anger': 0,
    'disgust': 1, 'disgusted': 1,
    'fear': 2, 'fearful': 2,
    'happy': 3, 'happiness': 3,
    'sad': 4, 'sadness': 4,
    'surprise': 5, 'suprise': 5,
    'neutral': 6, 'neutrality': 6,
    'contempt': 7
}

def download_data():
    print("--- Step 1: Downloading Data ---")
    downloaded_path = None
    for i in range(10):
        try:
            downloaded_path = kagglehub.dataset_download("arnabkumarroy02/ferplus")
            break
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(5)
    else:
        raise Exception("Download failed after multiple attempts")

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(downloaded_path), str(DATA_DIR))
    print(f"Data downloaded to: {DATA_DIR}")

def organize_images_and_create_csv():
    print("\n--- Step 2: Organizing Images & Creating CSV ---")
    
    if not DATA_DIR.exists():
        print(f"Directory not found: {DATA_DIR}")
        return

    if not DATA_RAW_DIR.exists():
        os.makedirs(DATA_RAW_DIR)

    data_rows = []
    files_to_process = []

    print(f"Scanning directory: {DATA_DIR}")
    for root, dirs, files in os.walk(DATA_DIR):
        if "raw" in Path(root).parts:
            continue

        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                files_to_process.append((root, filename))

    print(f"Found {len(files_to_process)} images.")

    for root, filename in tqdm(files_to_process, desc="Moving Images"):
        label_name = os.path.basename(root)
        label_id = EMOTION_MAPPING.get(label_name.lower(), -1)
        new_filename = f"{label_name}_{filename}"
        src_path = os.path.join(root, filename)
        dst_path = os.path.join(DATA_RAW_DIR, new_filename)
        shutil.move(src_path, dst_path)
        data_rows.append([new_filename, label_name, label_id])

    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label', 'label_id'])
        writer.writerows(data_rows)

    print(f"CSV saved at: {CSV_FILE}")

    print("Cleaning up original directories...")
    for folder in ['train', 'test', 'validation']:
        folder_path = DATA_DIR / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)

def make_typo(text):
    if len(text) > 2:
        idx = random.randint(0, len(text) - 1)
        return text[:idx] + text[idx + 1:]
    return text

def inject_noise():
    print("\n--- Step 3: Injecting Noise ---")
    if not CSV_FILE.exists():
        return

    random.seed(42)
    rows = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    count = 0
    for row in rows:
        if random.random() < 0.02:
            count += 1
            error = random.choice(['missing', 'typo', 'bad_id', 'noise'])
            if error == 'missing':
                if random.random() < 0.5: row['label'] = ""
                else: row['label_id'] = ""
            elif error == 'typo': row['label'] = make_typo(row['label'])
            elif error == 'bad_id': row['label_id'] = random.choice(['99', '-1', '100'])
            elif error == 'noise': row['label_id'] = 'NaN'

    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Modified {count} rows.")

def clean_data():
    print("\n--- Step 4: Cleaning Data ---")
    if not CSV_FILE.exists():
        return

    clean_rows = []
    removed_count = 0
    total_count = 0

    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            total_count += 1
            label = row.get('label', '')
            label_id = row.get('label_id', '')
            is_valid = True

            if not label or not label_id: is_valid = False
            elif label.lower() not in EMOTION_MAPPING: is_valid = False
            else:
                try:
                    lid = int(label_id)
                    if lid != EMOTION_MAPPING[label.lower()]: is_valid = False
                except ValueError: is_valid = False

            if is_valid: clean_rows.append(row)
            else: removed_count += 1

    with open(CLEANED_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)

    print(f"Cleaned data saved to: {CLEANED_CSV_FILE}")

def run_preparation():
    try:
        download_data()
        organize_images_and_create_csv()
        inject_noise()
        clean_data()
        print("\nAll data preparation steps completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred during preparation: {e}")

if __name__ == "__main__":
    run_preparation()