from ultralytics import YOLO
import os
import shutil
import random


def train_smart_bin():
    # 1. SETUP
    data_dir = os.path.abspath('garbage_dataset_white')

    # We need to ensure we have data in the 'val' (validation) folder too.
    # This simple snippet moves 20% of your training images to validation automatically.
    print("Organizing dataset...")
    for class_name in ['Restmuell', 'Biomuell', 'Gelber_Sack', 'Papiermuell']:
        train_path = os.path.join(data_dir, 'train', class_name)
        val_path = os.path.join(data_dir, 'val', class_name)

        images = os.listdir(train_path)
        # Move back existing val images to train first to reshuffle (optional, simpler for now)
        # Here we just check if val is empty, if so, move some images.
        if len(os.listdir(val_path)) < 2 and len(images) > 5:
            num_to_move = int(len(images) * 0.2)  # 20%
            to_move = random.sample(images, num_to_move)
            for img in to_move:
                shutil.move(os.path.join(train_path, img), os.path.join(val_path, img))

    # 2. LOAD MODEL
    # We load a pre-trained classification model
    model = YOLO('yolo11n-cls.pt')

    # 3. TRAIN
    # epochs=20 is usually enough for simple tasks.
    print("Starting training...")
    results = model.train(
        data=data_dir,
        epochs=20,
        imgsz=224,
        project='smart_bin_project',
        name='trash_model'
    )

    print(f"Training complete! Your new model is saved at: {results.save_dir}/weights/best.pt")
    print("Update the 'MODEL_PATH' in your smart_bin_app.py to point to this new file.")


if __name__ == '__main__':
    train_smart_bin()