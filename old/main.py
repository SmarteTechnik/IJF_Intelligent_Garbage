import cv2
import os
import time
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'smart_bin_project/trash_model_v1/weights/best.pt'  # Start with base model, change to 'runs/classify/train/weights/best.pt' after training
CONFIDENCE_THRESHOLD = 0.95  # 90% certainty required
DATASET_DIR = 'garbage_dataset'
CLASSES = ['Restmuell', 'Biomuell', 'Gelber_Sack', 'Papiermuell']

# --- SETUP DIRECTORIES ---
# Create folder structure: garbage_dataset/train/Restmuell, etc.
for split in ['train', 'val']:  # YOLO needs train and val folders
    for cls in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

# --- LOAD MODEL ---
# Note: On first run, this base model won't know your classes yet.
# It will be useful only after you run 'train_model.py' at least once.
try:
    model = YOLO(MODEL_PATH)
except:
    print("Model not found. Downloading base model...")
    model = YOLO('yolo11n-cls.pt')

# --- CAMERA LOOP ---
cap = cv2.VideoCapture(1)  # 0 is usually the default webcam

print(f"--- SMART BIN STARTED ---")
print(f"Press 'q' to quit.")
print(f"To label manually: 'r' (Rest), 'b' (Bio), 'g' (Gelb), 'p' (Papier)")


while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. PREDICT
    results = model(frame, verbose=False)
    probs = results[0].probs

    # Check if model has been trained on our custom classes yet
    if len(probs.data) == len(CLASSES):
        top1_index = probs.top1
        top1_conf = probs.top1conf.item()
        predicted_class = CLASSES[top1_index]
    else:
        # Fallback if using raw base model (before first training)
        top1_conf = 0.0
        predicted_class = "Unknown (Model untrained)"

    # 2. VISUALIZE
    # Show status on screen
    color = (0, 255, 0)  # Green by default
    status_text = f"Put into: {predicted_class} ({top1_conf:.1%})"

    # UNSURE LOGIC
    if top1_conf < CONFIDENCE_THRESHOLD:
        color = (0, 0, 255)  # Red
        status_text = "UNSURE: Please classify manually!"

    # Draw Text
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show probabilities for all classes
    if len(probs.data) == len(CLASSES):
        y_offset = 100
        for i, class_name in enumerate(CLASSES):
            prob = probs.data[i].item()
            text = f"{class_name}: {prob:.1%}"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30

    cv2.imshow('Smart Bin Camera', frame)

    # 3. USER INPUT (The "Buttons")
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Map keys to classes
    save_label = None
    if key == ord('r'):
        save_label = 'Restmuell'
    elif key == ord('b'):
        save_label = 'Biomuell'
    elif key == ord('g'):
        save_label = 'Gelber_Sack'
    elif key == ord('p'):
        save_label = 'Papiermuell'

    # 4. SAVE IMAGE FOR FINE-TUNING
    if save_label:
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}.jpg"
        # Save to 'train' folder by default
        save_path = os.path.join(DATASET_DIR, 'train', save_label, filename)
        cv2.imwrite(save_path, frame)
        print(f"Saved image to {save_label} for future training.")

cap.release()
cv2.destroyAllWindows()