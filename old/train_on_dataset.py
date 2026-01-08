from ultralytics import YOLO
import os
import shutil
import random

# --- CONFIGURATION ---
path = "garbage_dataset_white"
# path = "garbage_dataset"
DATA_DIR = os.path.abspath(path)
PROJECT_NAME = 'smart_bin_project'
# RUN_NAME = 'trash_model_v1'
RUN_NAME = 'trash_model_white_v1'


def prepare_data():
    """
    Moves 20% of images from 'train' to 'val' so YOLO can evaluate itself.
    """
    print("Checking dataset structure...")
    classes = ['Restmuell', 'Biomuell', 'Gelber_Sack', 'Papiermuell']

    for cls in classes:
        train_path = os.path.join(DATA_DIR, 'train', cls)
        val_path = os.path.join(DATA_DIR, 'val', cls)

        # Create val folder if it doesn't exist
        os.makedirs(val_path, exist_ok=True)

        # Check if we need to move files
        # We only move files if 'val' is empty and 'train' has images
        train_images = os.listdir(train_path)
        val_images = os.listdir(val_path)

        if len(val_images) == 0 and len(train_images) > 0:
            print(f"  Splitting data for {cls}...")
            num_to_move = int(len(train_images) * 0.2)  # 20% split
            files_to_move = random.sample(train_images, num_to_move)

            for f in files_to_move:
                shutil.move(os.path.join(train_path, f), os.path.join(val_path, f))
        else:
            print(f"  {cls}: Ready (Train: {len(train_images)}, Val: {len(val_images)})")


def train():
    # 1. Prepare the folders
    prepare_data()

    # 2. Load the base model (nano version is fastest)
    # The first time you run this, it will download yolo11n-cls.pt
    model = YOLO('yolo11n-cls.pt')
    model.to('mps')

    # 3. Start Training
    print("\n--- STARTING TRAINING ---")
    print("This may take a while depending on your computer speed...")

    results = model.train(
        data=DATA_DIR,
        epochs=10,  # 10 is usually enough for this dataset
        imgsz=224,  # Standard size for classification
        batch=256,  # Reduce to 8 if you run out of memory
        project=PROJECT_NAME,
        name=RUN_NAME
    )

    # 4. Success Message
    print("\n" + "=" * 40)
    print("TRAINING FINISHED!")
    print(f"Your new intelligent model is saved here:")
    print(f"{results.save_dir}/weights/best.pt")
    print("=" * 40)


if __name__ == '__main__':
    train()