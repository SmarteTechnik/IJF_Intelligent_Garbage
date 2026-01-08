import os

# Kaggle Credentials (REPLACE WITH YOURS)
K_USERNAME = "jonathanpi27"
K_KEY = "KGAT_f165dd17569ca42ed64b93b273da648b"
os.environ['KAGGLE_USERNAME'] = K_USERNAME
os.environ['KAGGLE_KEY'] = K_KEY

import shutil
import zipfile
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from kaggle.api.kaggle_api_extended import KaggleApi

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
DATASET_NAME = "viswaprakash1990/garbage-detection"
ZIP_FILE = "garbage-detection.zip"
DATA_ROOT = "smart_bin_detection_data"
PROJECT_NAME = "smart_bin_project"
RUN_NAME = "german_trash_det_v1"

# Map the Dataset ID (0-5) to German Names
# Dataset Classes: 0:Biodegradable, 1:Cardboard, 2:Glass, 3:Metal, 4:Paper, 5:Plastic
CLASS_MAPPING = {
    0: "Biomuell (Bio)",
    1: "Papiermuell (Cardboard)",
    2: "Altglas (Glass)",
    3: "Gelber Sack (Metal)",
    4: "Papiermuell (Paper)",
    5: "Gelber Sack (Plastic)"
}


# ==========================================
# 2. DATASET PREPARATION (NATIVE)
# ==========================================
def create_data_yaml(root_path):
    """
    Creates the 'data.yaml' file required by YOLO Detection.
    It tells YOLO where the images are and what the names are.
    """
    yaml_path = os.path.join(root_path, "data.yaml")

    # We must point to absolute paths for safety
    abs_path = os.path.abspath(root_path)

    data = {
        'path': abs_path,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 6,  # Number of classes in the raw dataset
        'names': ['Biodegradable', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic']
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    return yaml_path


def setup_dataset():
    print("\n--- STEP 1: CHECKING DATASET ---")
    if os.path.exists(DATA_ROOT) and os.path.exists(os.path.join(DATA_ROOT, "data.yaml")):
        print("Dataset ready.")
        return os.path.join(DATA_ROOT, "data.yaml")

    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(ZIP_FILE):
        print(f"Downloading {DATASET_NAME}...")
        api.dataset_download_files(DATASET_NAME, unzip=False, path=".", quiet=False)

    # Extract to a temp folder first to check structure
    temp_dir = "temp_raw"
    if not os.path.exists(temp_dir):
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

    # Move content to DATA_ROOT
    # The zip usually contains a subfolder named "Garbage detection" or similar
    # We want to move the inner 'train', 'valid', 'test' folders to DATA_ROOT
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    # Locate where the 'train' folder is inside the unzipped path
    target_dir = temp_dir
    for root, dirs, files in os.walk(temp_dir):
        if "train" in dirs and "valid" in dirs:
            target_dir = root
            break

    # Move files
    for split in ["train", "valid", "test"]:
        shutil.move(os.path.join(target_dir, split), os.path.join(DATA_ROOT, split))

    # Clean up
    shutil.rmtree(temp_dir)

    # Create the config file
    yaml_path = create_data_yaml(DATA_ROOT)
    print("Dataset setup complete.")
    return yaml_path


# ==========================================
# 3. TRAINING LOOP (DETECTION)
# ==========================================
def train_or_load_model(yaml_path):
    print("\n--- STEP 2: CHECKING MODEL ---")
    model_path = f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"

    if os.path.exists(model_path):
        print(f"Found model: {model_path}")
        model = YOLO(model_path)
        model.to('mps')
        return model

    print("Starting Object Detection Training...")
    # NOTICE: We use 'yolo11n.pt' (Detection), NOT 'yolo11n-cls.pt'
    model = YOLO('yolo11n.pt')
    model.to('mps')

    model.train(
        data=yaml_path,
        epochs=8,
        imgsz=640,
        project=PROJECT_NAME,
        name=RUN_NAME,
        batch=16,
        max_det=3,
        workers=6,
        half=True,
        verbose=True
    )
    trained_model = YOLO(model_path)
    trained_model.to('mps')
    return trained_model


# ==========================================
# 4. VISUALIZATION
# ==========================================
def draw_stats_window(detections):
    """
    Draws a bar chart of the DETECTED objects.
    """
    h, w = 400, 500
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 40

    cv2.putText(canvas, "Live Detections", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if not detections:
        cv2.putText(canvas, "No Trash Detected", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 1)
        return canvas

    # Aggregate results (e.g. if 2 plastic bottles, show Plastic once with max conf)
    found_classes = {}
    for cls_id, conf in detections:
        # Use our German mapping
        german_name = CLASS_MAPPING.get(cls_id, "Unknown")
        if german_name not in found_classes or conf > found_classes[german_name]:
            found_classes[german_name] = conf

    y_pos = 100
    for name, conf in found_classes.items():
        # Draw Bar
        bar_w = int(conf * 300)
        cv2.putText(canvas, name, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.rectangle(canvas, (20, y_pos + 10), (20 + bar_w, y_pos + 30), (0, 255, 0), -1)
        cv2.putText(canvas, f"{conf:.1%}", (25 + bar_w, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 60

    return canvas


# ==========================================
# 5. MAIN
# ==========================================
def main():
    yaml_path = setup_dataset()
    model = train_or_load_model(yaml_path)

    cap = cv2.VideoCapture(1)  # Check camera ID

    print("Running Smart Bin (Detection Mode)...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Inference
        results = model(frame, verbose=False)
        result = results[0]

        display_frame = frame.copy()
        current_detections = []  # Store (class_id, conf)

        # Iterate over detected boxes
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf < 0.2: continue  # Skip low confidence

            current_detections.append((cls_id, conf))

            # Get German Label
            german_label = CLASS_MAPPING.get(cls_id, "Restmuell")

            # Draw Box on Camera Feed
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{german_label} {conf:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show Windows
        cv2.imshow("Smart Bin Camera", display_frame)

        stats = draw_stats_window(current_detections)
        cv2.imshow("Stats", stats)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()