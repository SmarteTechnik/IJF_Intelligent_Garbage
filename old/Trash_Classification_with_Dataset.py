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
import random
from ultralytics import YOLO
from kaggle.api.kaggle_api_extended import KaggleApi

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
# Dataset Config
DATASET_NAME = "mostafaabla/garbage-classification"  # Contains Clothes/Bio/Shoes
ZIP_FILE = "garbage-classification.zip"
DATA_ROOT = "smart_bin_data"
PROJECT_NAME = "smart_bin_project"
RUN_NAME = "german_trash_model"

# German Categories Mapping
# We map the English dataset folders to your 6 specific German bins
CATEGORY_MAPPING = {
    # Papiermuell (Blaue Tonne)
    "paper": "Papiermuell",
    "cardboard": "Papiermuell",

    # Gelber Sack (Plastic/Metal)
    "plastic": "Gelber_Sack",
    "metal": "Gelber_Sack",

    # Biomuell (Brown)
    "biological": "Biomuell",

    # Altglas (Glass Container)
    "brown-glass": "Altglas",
    "green-glass": "Altglas",
    "white-glass": "Altglas",

    # Altkleider (Clothes/Shoes)
    "clothes": "Altkleider",
    "shoes": "Altkleider",

    # Restmuell (Everything else)
    "trash": "Restmuell",
    "battery": "Restmuell"
}

CLASSES = sorted(list(set(CATEGORY_MAPPING.values())))  # ['Altglas', 'Altkleider', 'Biomuell', ...]


# ==========================================
# 2. DATASET PREPARATION
# ==========================================
def setup_dataset():
    print("\n--- STEP 1: CHECKING DATASET ---")

    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir = os.path.join(DATA_ROOT, "val")

    # Check if data already exists and is organized
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) >= 6:
        print("Dataset already organized. Skipping download.")
        return

    # Authenticate Kaggle
    os.environ['KAGGLE_USERNAME'] = K_USERNAME
    os.environ['KAGGLE_KEY'] = K_KEY
    api = KaggleApi()
    api.authenticate()

    # Download
    if not os.path.exists(ZIP_FILE):
        print(f"Downloading dataset: {DATASET_NAME}...")
        api.dataset_download_files(DATASET_NAME, unzip=False, path=".")

    # Extract
    temp_extract_dir = "temp_raw_data"
    if not os.path.exists(temp_extract_dir):
        print("Extracting zip file...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)

    # Organize
    print("Organizing into German categories...")

    # Find internal root (some zips have nested folders)
    base_path = temp_extract_dir
    for root, dirs, files in os.walk(temp_extract_dir):
        if "plastic" in dirs:
            base_path = root
            break

    # Create directories
    for split in ["train", "val"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(DATA_ROOT, split, cls), exist_ok=True)

    # Move and Split files
    for eng_name, ger_name in CATEGORY_MAPPING.items():
        source_path = os.path.join(base_path, eng_name)
        if not os.path.exists(source_path):
            continue

        all_files = os.listdir(source_path)
        random.shuffle(all_files)

        # 80% Train, 20% Val
        split_idx = int(len(all_files) * 0.8)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        for f in train_files:
            shutil.move(os.path.join(source_path, f), os.path.join(DATA_ROOT, "train", ger_name, f))
        for f in val_files:
            shutil.move(os.path.join(source_path, f), os.path.join(DATA_ROOT, "val", ger_name, f))

    # Cleanup
    shutil.rmtree(temp_extract_dir)
    print("Dataset setup complete.")


# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_or_load_model():
    print("\n--- STEP 2: CHECKING MODEL ---")

    # Check if model exists
    model_path = f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"

    if os.path.exists(model_path):
        print(f"Found existing trained model at: {model_path}")
        model = YOLO(model_path)
        model.to("mps")
        return model

    print("No trained model found. Starting training...")
    model = YOLO('yolo11n-cls.pt')  # Load base model
    model.to("mps")
    model.train(
        data=os.path.abspath(DATA_ROOT),
        epochs=10,
        imgsz=224,
        project=PROJECT_NAME,
        name=RUN_NAME,
        batch=128,
        verbose=True,
        degrees=10,  # Rotate images slightly (+/- 10 degrees)
        hsv_h=0.015,  # Adjust color hue (makes it ignore exact "red" vs "orange")
        hsv_s=0.4,  # Adjust saturation (helps with lighting differences)
        hsv_v=0.4,  # Adjust brightness (helps with shadows)
        scale=0.5,  # Zoom in/out (helps if object size varies)
        fliplr=0.5,  # Flip left/right
        mosaic=1.0,  # Mixes images together (very strong for YOLO)
    )
    print("Training finished.")
    return YOLO(model_path)  # Reload the best weights


# ==========================================
# 4. IMAGE PROCESSING (WHITE BACKGROUND)
# ==========================================
def replace_background_with_white(frame, bg_model):
    """
    Subtracts the static background and replaces it with white.
    Returns: Processed Frame (White BG), Mask
    """
    # Calculate absolute difference between current frame and background
    diff = cv2.absdiff(bg_model, frame)

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # Clean up noise (Morphological operations)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Create white background
    white_bg = np.full_like(frame, 255)

    # Combine: Where mask is black (background), use white_bg.
    # Where mask is white (object), use original frame.
    # Invert mask for background logic
    mask_inv = cv2.bitwise_not(mask)

    object_part = cv2.bitwise_and(frame, frame, mask=mask)
    white_part = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)

    final_result = cv2.add(object_part, white_part)
    return final_result, mask


# ==========================================
# 5. MAIN APPLICATION
# ==========================================
def main():
    # A. Setup
    setup_dataset()
    model = train_or_load_model()

    # B. Camera Init
    cap = cv2.VideoCapture(1)

    print("\n--- SMART BIN LIVE FEED ---")
    print("CONTROLS:")
    print(" [B] - Capture Background (Remove all items first!)")
    print(" [W] - Toggle White Background Mode (On/Off)")
    print(" [Q] - Quit")

    use_white_bg = False
    background_model = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        inference_frame = frame.copy()

        # C. Background Logic
        if use_white_bg and background_model is not None:
            processed_frame, mask = replace_background_with_white(frame, background_model)
            inference_frame = processed_frame  # The AI sees the white-bg version

            # Show small preview of what the AI sees in corner
            small_preview = cv2.resize(inference_frame, (160, 120))
            display_frame[0:120, 0:160] = small_preview
            cv2.putText(display_frame, "AI Input", (5, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # D. Prediction
        results = model(inference_frame, verbose=False)
        probs = results[0].probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = results[0].names[top1_idx]

        # E. Visualization
        # Color coding
        color = (0, 255, 0)
        if top1_conf < 0.5: color = (0, 0, 255)  # Red if unsure

        label = f"{class_name}: {top1_conf:.1%}"
        cv2.putText(display_frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Status of Modes
        mode_text = "Mode: Normal"
        if use_white_bg: mode_text = "Mode: White BG (Active)"
        if use_white_bg and background_model is None: mode_text = "Mode: White BG (NO BG SET! Press B)"

        cv2.putText(display_frame, mode_text, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0),
                    2)

        cv2.imshow('Smart Bin', display_frame)

        # F. User Inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            print("Background captured! Please put item in frame now.")
            background_model = frame.copy()
            # Slight blur to reduce camera noise
            background_model = cv2.GaussianBlur(background_model, (5, 5), 0)
        elif key == ord('w'):
            use_white_bg = not use_white_bg
            print(f"White background mode: {use_white_bg}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()