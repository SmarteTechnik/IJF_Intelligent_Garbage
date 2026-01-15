import os
import shutil
import sys
import cv2
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSpacerItem, QProgressBar)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO


class TrashClassificatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model_ready = False
        self.epochs_to_train = 10
        self.trainer = None

        # --- UI Configuration ---
        self.target_height = 400  # Define your specific height here
        self.setWindowTitle("Intelligenter Müll Trenner")

        self.categories = ["Biomüll", "Gelber Sack", "Papiermüll", "Restmüll"]

        # 1. Video Label (Left Side)
        self.video_label = QLabel("Stream Loading...")
        self.video_label.setFixedHeight(self.target_height)
        self.video_label.setStyleSheet("background-color: black;")  # Visual placeholder

        # 2. Buttons (Right Side)
        self.sidebar_layout = QVBoxLayout()
        self.buttons = []
        self.prediction_labels = []

        title_label = QLabel(f"<h1>Vorhersagen des Models</h1>")
        title_label.setWordWrap(True)
        self.sidebar_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)


        for category in self.categories:
            self.add_prediction_label(category)

        finetune_label = QLabel(f"<h3>Nachtrainieren</h3>"
                                f"<p>Falls die Vorhersage des Models falsch ist,<br>klicke die Kategorie an, zu der der Abfall gehört."
                                f"Dann wird davon ein Bild gespeichert.Falls du das Model dann mit den zusätzlich gemachten"
                                f"Bildern trainieren möchtest, drücke unten auf den Button<br><b>Nachtrainieren</b>.</p>")
        finetune_label.setWordWrap(True)
        self.sidebar_layout.addWidget(finetune_label,  alignment=Qt.AlignmentFlag.AlignCenter)

        self.add_button(self.categories[0], self.save_image, "b", "LawnGreen")
        self.add_button(self.categories[1], self.save_image, "g", "yellow")
        self.add_button(self.categories[2], self.save_image, "p", "aqua")
        self.add_button(self.categories[3], self.save_image, "r", "LightGray")
        # Add a spacer to push buttons to the top (optional)
        self.sidebar_layout.addStretch()
        verticalSpacer = QSpacerItem(20, 40)
        self.sidebar_layout.addItem(verticalSpacer)
        self.add_button("Nachtrainieren", self.finetune_model, color="white")
        self.buttons[-1].setStyleSheet("background-color: white; font-weight: bold")

        # 3. Main Horizontal Layout
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.video_label)  # Added first = Left
        self.main_layout.addLayout(self.sidebar_layout)  # Added second = Right

        # --- Process Bar ---
        self.pbar = QProgressBar(self)
        self.pbar.setRange(1, self.epochs_to_train)
        self.pbar.setValue(1)
        self.pbar.setVisible(False)

        self.sidebar_layout.addWidget(self.pbar)
        self.setLayout(self.main_layout)

        # --- OpenCV Logic ---
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # --- Model ---
        self.DATA_DIR = os.path.join("DATA")
        self.STD_SOURCE_DIR = os.path.join(self.DATA_DIR, "std_own_data")
        self.CUSTOM_SOURCE_DIR = os.path.join(self.DATA_DIR, "custom_own_data")
        self.STD_DATASET_DIR = os.path.join(self.DATA_DIR, "std_own_yolo_dataset")
        self.CUSTOM_DATASET_DIR = os.path.join(self.DATA_DIR, "custom_own_yolo_dataset")
        self.PROJECT_NAME = os.path.join(self.DATA_DIR, "trash_classification")
        self.STD_RUN_NAME = "std_trash_model"
        self.CUSTOM_RUN_NAME = "custom_trash_model"

        self.std_weights_path = os.path.join(self.PROJECT_NAME, self.STD_RUN_NAME, "weights", "best.pt")
        self.custom_weights_path = os.path.join(self.PROJECT_NAME, self.CUSTOM_RUN_NAME, "weights", "best.pt")
        if not os.path.exists(self.std_weights_path):
            self.setup_std_model()
        print("Loading YOLO Model...")
        self.model = YOLO(self.std_weights_path)
        self.model.to('mps')
        self.model_ready = True
        print("Model loaded. Ready to predict.")

        self.prepare_custum_model()

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w

            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            # Scale the pixmap to your specific height while keeping aspect ratio
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaledToHeight(self.target_height, Qt.TransformationMode.SmoothTransformation)

            self.video_label.setPixmap(scaled_pixmap)
            if self.model_ready:
                results = self.model(frame, verbose=False)
                pred = results[0].probs.data.cpu().numpy()
                class_names = results[0].names
                for label, prediction, name_idx in zip(self.prediction_labels, pred, class_names):
                    label.setText(f"<u>{class_names[name_idx]}:</u> <strong>{float(prediction):.2f}%</strong>")

                top_pred = pred.argmax()
                for i, p in enumerate(pred):
                    if i == top_pred:
                        self.prediction_labels[top_pred].setStyleSheet("background-color: aquamarine;")
                    else:
                        self.prediction_labels[i].setStyleSheet("")




    def add_button(self, name, function, arg=None, color="white"):
        btn = QPushButton(f"{name}")
        btn.setFixedWidth(160)  # Keep buttons uniform
        if arg is not None:
            btn.pressed.connect(lambda: function(arg))
        else:
            btn.pressed.connect(function)
        btn.setStyleSheet(f"background-color: {color};")
        self.sidebar_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.buttons.append(btn)

    def add_prediction_label(self, name):
        label = QLabel(f"<u>{name}:</u> <strong>0.00%</strong>")
        self.sidebar_layout.addWidget(label)
        self.prediction_labels.append(label)

    def closeEvent(self, event):
        self.capture.release()

    def setup_std_model(self):
        """
        Create from image folder a Dataset-Folder for YOLO8
        :return: None
        """
        print("Setting up Standard Model...")
        # --- Check Folder Structure ---
        data_folder_names = [
            os.path.join(self.STD_SOURCE_DIR, "Biomuell"),
            os.path.join(self.STD_SOURCE_DIR, "Gelber_Sack"),
            os.path.join(self.STD_SOURCE_DIR, "Papiermuell"),
            os.path.join(self.STD_SOURCE_DIR, "Restmuell"),
        ]
        error = False
        for folder in data_folder_names:
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
                print(f"Creating folder: {folder}")
                error = True
        if error:
            raise Exception("Fehler: Leider wurden keine Anfangsbilder hinzugefügt. Das Model ist nicht einsatzbereit. "
                            f"Bitte lege für jede Kategorie ein Bild ab. Diese sollen in {os.path.join(self.DATA_DIR, self.STD_SOURCE_DIR)}\n"
                            f"Die Ordner dafür wurden bereits erstellt.")

        # --- Prepare YOLO Dataset ---
        print("Preparing YOLO Dataset...")
        self.create_dataset(self.STD_DATASET_DIR, self.STD_SOURCE_DIR)
        print(f"Beginning the training of the model")
        model = YOLO('yolo11n-cls.pt')

        # Train the model
        # We allow it to overwrite the previous project folder (exist_ok=True)
        # so we don't get quick_shot_run2, quick_shot_run3, etc.
        model.train(
            data=self.STD_DATASET_DIR,
            epochs=25,
            imgsz=640,
            project=str(self.PROJECT_NAME),
            name=str(self.STD_RUN_NAME),
            exist_ok=True
        )
        print("Training complete.")
        self.model_ready = True

    def create_dataset(self, dataset_folder_path, image_data_path):
        if os.path.exists(dataset_folder_path):
            shutil.rmtree(dataset_folder_path)

        modes = ['train', 'val']
        data_folders = [d for d in os.listdir(image_data_path) if
                        os.path.isdir(os.path.join(image_data_path, d))]

        for mode in modes:
            for cat in data_folders:
                dest_path = os.path.join(dataset_folder_path, mode, cat)
                os.makedirs(dest_path, exist_ok=True)

                source_cat_path = str(os.path.join(image_data_path, cat))
                for img_file in os.listdir(source_cat_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        shutil.copy(
                            os.path.join(source_cat_path, img_file),
                            os.path.join(dest_path, img_file)
                        )
        print("Data ready.")

    def prepare_custum_model(self):
        if os.path.exists(self.CUSTOM_SOURCE_DIR):
            shutil.rmtree(self.CUSTOM_SOURCE_DIR)

            # 2. Copy the entire tree
        shutil.copytree(self.STD_SOURCE_DIR, self.CUSTOM_SOURCE_DIR)
        print(f"Eigenes Verzeichnis für Nachtrainierungsdaten in '{self.CUSTOM_SOURCE_DIR}' "
              f"angelegt mit Grunddaten von '{self.STD_SOURCE_DIR}'")

    def save_image(self, category):
        print(f"Saving image for category '{category}'...")
        ret, frame = self.capture.read()
        if category == 'b':
            filename = suggest_next_filename(
                os.path.join(self.CUSTOM_SOURCE_DIR, "Biomuell")
            )
        elif category == 'g':
            filename = suggest_next_filename(
                os.path.join(self.CUSTOM_SOURCE_DIR, "Gelber_Sack")
            )
        elif category == 'p':
            filename = suggest_next_filename(
                os.path.join(self.CUSTOM_SOURCE_DIR, "Papiermuell")
            )
        elif category == 'r':
            filename = suggest_next_filename(
                os.path.join(self.CUSTOM_SOURCE_DIR, "Restmuell")
            )
        else:
            raise Exception(f"Unbekannte Kategorie: '{category}'")

        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")


    def finetune_model(self):
        print("Nachtrainieren des Modells...")
        self.model_ready = False
        self.create_dataset(self.CUSTOM_DATASET_DIR, self.CUSTOM_SOURCE_DIR)
        del self.model
        self.trainer = TrainerThread(parent=self)
        self.trainer.progress_update.connect(self.pbar.setValue)
        self.trainer.start()

    def custom_train_finished(self):
        self.model = YOLO(self.custom_weights_path)
        self.model_ready = True


def suggest_next_filename(folder_path, extension=".jpg"):
    """
    Scans the folder for files named '1.png', '2.png', etc.
    Returns the full path for the next number in the sequence.
    """
    # Ensure folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return os.path.join(folder_path, f"1{extension}")

    max_number = 0

    # Iterate over all files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(extension):
            # Strip extension and check if the rest is a number
            name_without_ext = filename[:-len(extension)]

            if name_without_ext.isdigit():
                current_number = int(name_without_ext)
                if current_number > max_number:
                    max_number = current_number

    # Calculate next number
    next_name = f"{max_number + 1}{extension}"
    return os.path.join(folder_path, next_name)


class TrainerThread(QThread):
    # Signal to send the current epoch number to the GUI
    progress_update = pyqtSignal(int)

    def __init__(self, parent: TrashClassificatorApp = None):
        super().__init__()
        self.c_model = YOLO("yolo11n-cls.pt")
        self.parent = parent

    def on_train_epoch_end(self, trainer):
        # trainer.epoch is the current index, trainer.epochs is total
        # We send the current epoch + 1 to the GUI
        current_epoch = trainer.epoch + 1
        self.progress_update.emit(current_epoch)

    def run(self):
        # Add the callback to the model
        self.c_model.add_callback("on_train_epoch_end", self.on_train_epoch_end)
        self.parent.pbar.setVisible(True)
        # Start training
        self.c_model.train(data=self.parent.CUSTOM_DATASET_DIR,
        epochs=self.parent.epochs_to_train,
        imgsz=640,
        project=str(self.parent.PROJECT_NAME),
        name=str(self.parent.CUSTOM_RUN_NAME),
        exist_ok=True)
        self.parent.pbar.setVisible(False)
        self.parent.custom_train_finished()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrashClassificatorApp()
    window.show()
    sys.exit(app.exec())