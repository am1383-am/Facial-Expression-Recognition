import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(csv_path, image_dir, batch_size=64, target_size=(48, 48)):
    df = pd.read_csv(csv_path)
    df['filename'] = df['filename'].astype(str)
    df['label'] = df['label'].astype(str)

    train_val_df, test_df = train_test_split(
        df, 
        test_size=0.10, 
        random_state=42, 
        stratify=df['label']
    )
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.11,
        random_state=42, 
        stratify=train_val_df['label']
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    common_args = {
        "directory": image_dir,
        "x_col": "filename",
        "y_col": "label",
        "target_size": target_size,
        "batch_size": batch_size,
        "class_mode": "categorical",
        "color_mode": "grayscale"
    }

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        subset=None,
        shuffle=True,
        **common_args
    )

    val_gen = test_datagen.flow_from_dataframe(
        dataframe=val_df,
        shuffle=False,
        **common_args
    )
    
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        shuffle=False,
        **common_args
    )

    return train_gen, val_gen, test_gen