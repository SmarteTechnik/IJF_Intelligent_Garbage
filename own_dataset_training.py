import os
import shutil
import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
SOURCE_DIR = "std_own_data"
DATASET_DIR = "own_yolo_dataset"
PROJECT_NAME = "own_trash_classification"
RUN_NAME = "quick_shot_run"


def prepare_data():
    """Duplicates images into train/val folders so YOLO can run."""
    print("--- Step 1: Preparing Data ---")

    # Clean up old dataset folder if it exists
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    modes = ['train', 'val']
    categories = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

    if not categories:
        print("Error: No category folders found in 'own_data'!")
        return False

    for mode in modes:
        for cat in categories:
            dest_path = os.path.join(DATASET_DIR, mode, cat)
            os.makedirs(dest_path, exist_ok=True)

            source_cat_path = os.path.join(SOURCE_DIR, cat)
            for img_file in os.listdir(source_cat_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy(
                        os.path.join(source_cat_path, img_file),
                        os.path.join(dest_path, img_file)
                    )
    print("Data ready.")
    return True


def train_model():
    """Trains the YOLO classification model."""
    print("--- Step 2: Training Model ---")

    # Load a pretrained YOLOv11n classification model
    model = YOLO('yolo11n-cls.pt')

    # Train the model
    # We allow it to overwrite the previous project folder (exist_ok=True)
    # so we don't get quick_shot_run2, quick_shot_run3, etc.
    model.train(
        data=DATASET_DIR,
        epochs=25,
        imgsz=640,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True
    )
    print("Training complete.")

    # Return the path to the best saved model weights
    best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "best.pt")
    return best_model_path


def run_live_prediction(model_path):
    """Runs the webcam and predicts using the trained model."""
    print(f"--- Step 3: Starting Live Prediction using {model_path} ---")

    # Load the CUSTOM trained model
    model = YOLO(model_path)

    cap = cv2.VideoCapture(1)  # 0 is usually the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        results = model(frame, verbose=False)

        # 'plot()' automatically draws the classification results/probabilities on the frame
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Trash Classifier", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 1. Prepare
    if prepare_data():
        # 2. Train
        best_weights = train_model()

        # 3. Predict
        # Check if training actually produced the file
        if os.path.exists(best_weights):
            run_live_prediction(best_weights)
        else:
            print("Something went wrong. Could not find the trained model file.")