import sys
import os
import shutil
import cv2
import torch
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSpacerItem, QProgressBar, QMessageBox)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QImage, QPixmap, QDesktopServices
from ultralytics import YOLO

# --- PATH FIX FOR FROZEN APP ---
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle/exe, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    # We want the working directory to be where the executable is,
    # so we can find the 'DATA' folder next to it.
    application_path = os.path.dirname(sys.executable)
    os.chdir(application_path)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

# --- Konfiguration ---
TARGET_HEIGHT = 400
SMOOTHING_ALPHA = 0.05  # Glättungsfaktor für Vorhersagen (0.1 = weich, 0.9 = reaktiv)
EPOCHS_TO_TRAIN = 15
CATEGORIES = ["Gelber Sack", "Papiermüll", "Restmüll"]
CLASSIFICATION_MODEL = "yolo26n-cls.pt"


def get_best_device():
    """Ermittelt das beste verfügbare Gerät für PyTorch (CUDA, MPS oder CPU)."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


DEVICE = get_best_device()
print(f"Nutze Gerät für Berechnungen: {DEVICE.upper()}")


class TrainerThread(QThread):
    """
    Hintergrund-Thread für das Nachtrainieren des YOLO-Modells,
    damit die GUI während des Trainings nicht einfriert.
    """
    progress_update = pyqtSignal(int)
    training_finished = pyqtSignal()

    def __init__(self, dataset_path, project_path, run_name, epochs):
        super().__init__()
        self.dataset_path = dataset_path
        self.project_path = project_path
        self.run_name = run_name
        self.epochs = epochs
        self.model = None

    def on_train_epoch_end(self, trainer):
        """Callback von YOLO, wird am Ende jeder Epoche aufgerufen."""
        current_epoch = trainer.epoch + 1
        self.progress_update.emit(current_epoch)

    def run(self):
        # Initialisiere ein frisches Modell für das Transfer-Learning
        self.model = YOLO(CLASSIFICATION_MODEL)

        # Callback registrieren
        self.model.add_callback("on_train_epoch_end", self.on_train_epoch_end)

        # Training starten
        res = self.model.train(
            data=str(self.dataset_path),
            epochs=self.epochs,
            imgsz=640,
            project=str(self.project_path),
            name=self.run_name,
            device=DEVICE,
            exist_ok=True,
            verbose=False
        )

        # Sicherstellen, dass die Gewichte korrekt kopiert werden
        res_save_weights = (Path(res.save_dir) / "weights" / "last.pt").absolute()
        target_weights_dir = Path(self.project_path) / self.run_name / "weights"
        target_weights_file = (target_weights_dir / "last.pt").absolute()

        target_weights_dir.mkdir(parents=True, exist_ok=True)

        if res_save_weights.exists() and res_save_weights != target_weights_file:
            shutil.copy2(res_save_weights, target_weights_file)
            print(f"Neues Modell gespeichert unter: {target_weights_file}")

        self.training_finished.emit()


class TrashClassificatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligenter Müll-Trenner")

        # Kamera Einstellungen
        self.current_cam_index = 0
        self.capture = None

        # Status-Variablen
        self.model_ready = False
        self.trainer = None
        self.current_probs = None

        # Pfad-Initialisierung (Verwendung von pathlib für OS-Unabhängigkeit)
        self.base_dir = Path("DATA")
        self.std_source_dir = self.base_dir / "std_own_data"
        self.custom_source_dir = self.base_dir / "custom_own_data"
        self.std_dataset_dir = self.base_dir / "std_own_yolo_dataset"
        self.custom_dataset_dir = self.base_dir / "custom_own_yolo_dataset"
        self.project_path = self.base_dir / "trash_classification"

        self.std_run_name = "std_trash_model"
        self.custom_run_name = "custom_trash_model"

        self.std_weights_path = self.project_path / self.std_run_name / "weights" / "last.pt"
        self.custom_weights_path = self.project_path / self.custom_run_name / "weights" / "last.pt"

        # UI Aufbauen
        self.init_ui()
        self.initialize_camera()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Modell initialisieren
        self.initialize_model_system()

    def init_ui(self):
        """Erstellt und platziert alle UI-Elemente."""
        self.main_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.sidebar_layout = QVBoxLayout()

        # --- Linke Seite: Video & Nachrichten ---
        self.video_label = QLabel("Video wird geladen...")
        self.video_label.setFixedHeight(TARGET_HEIGHT)
        self.video_label.setStyleSheet("background-color: black; color: white; qproperty-alignment: AlignCenter;")
        self.left_layout.addWidget(self.video_label)

        self.message_label = QLabel("")
        self.message_label.setMinimumWidth(400)
        self.message_label.setWordWrap(True)
        self.left_layout.addWidget(self.message_label, alignment=Qt.AlignmentFlag.AlignLeft)

        # --- Rechte Seite: Steuerung & Vorhersagen ---
        self.switch_cam_btn = QPushButton(f"Kamera wechseln (Aktuell: {self.current_cam_index})")
        self.switch_cam_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold;")
        self.switch_cam_btn.pressed.connect(self.switch_camera)
        self.sidebar_layout.addWidget(self.switch_cam_btn)
        title_label = QLabel("<h1>Vorhersage</h1>")
        self.sidebar_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.prediction_labels = []
        for category in CATEGORIES:
            lbl = QLabel(f"<u>{category}:</u> <strong>0.00%</strong>")
            self.sidebar_layout.addWidget(lbl)
            self.prediction_labels.append(lbl)

        finetune_info = QLabel("<h3>Nachtrainieren</h3>"
                               "<p>Ist die Vorhersage falsch?<br>"
                               "Klicke auf die korrekte Kategorie, um ein Bild zu speichern.<br>"
                               "Klicke danach auf 'Nachtrainieren'.</p>")
        finetune_info.setWordWrap(True)
        self.sidebar_layout.addWidget(finetune_info, alignment=Qt.AlignmentFlag.AlignCenter)

        # Buttons erstellen
        self.buttons: list[QPushButton] = []
        # Gelber Sack (Gelb)
        self.create_button("Gelber Sack", lambda: self.save_training_image("Gelber_Sack"), "yellow")
        # Papiermüll (Blau/Cyan)
        self.create_button("Papiermüll", lambda: self.save_training_image("Papiermuell"), "aqua")
        # Restmüll (Grau)
        self.create_button("Restmüll", lambda: self.save_training_image("Restmuell"), "lightgray")

        self.sidebar_layout.addStretch()

        self.train_btn = self.create_button("Nachtrainieren", self.start_finetuning, "white")
        self.train_btn.setStyleSheet("background-color: white; font-weight: bold; padding: 5px;")
        self.buttons.append(self.train_btn)

        self.sidebar_layout.addStretch()
        self.open_folder_btn = QPushButton("Trainingsbilder anzeigen")
        self.open_folder_btn.setStyleSheet("background-color: white; color: black; font-style: italic;")
        self.open_folder_btn.pressed.connect(self.open_data_folder)
        self.sidebar_layout.addWidget(self.open_folder_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Fortschrittsbalken
        self.pbar = QProgressBar()
        self.pbar.setRange(0, EPOCHS_TO_TRAIN)
        self.pbar.setValue(0)
        self.pbar.setVisible(False)
        self.sidebar_layout.addWidget(self.pbar)

        # Layouts zusammenfügen
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.sidebar_layout)
        self.setLayout(self.main_layout)

    def initialize_camera(self):
        """Initialisiert die Kamera mit dem korrekten Backend für macOS."""
        if self.capture is not None:
            self.capture.release()

        print(f"Versuche Kamera-Index: {self.current_cam_index}")

        # WICHTIGER FIX FÜR MACOS: cv2.CAP_AVFOUNDATION explizit nutzen
        if sys.platform == "darwin":
            self.capture = cv2.VideoCapture(self.current_cam_index, cv2.CAP_AVFOUNDATION)
        else:
            self.capture = cv2.VideoCapture(self.current_cam_index)

        # Prüfen ob erfolgreich
        if not self.capture.isOpened():
            self.video_label.setText(f"Fehler: Kamera {self.current_cam_index} nicht gefunden.\n"
                                     "Bitte 'Kamera wechseln' klicken.")
        else:
            # Ein Frame lesen zum Testen
            ret, _ = self.capture.read()
            if not ret:
                self.video_label.setText(f"Kamera {self.current_cam_index} verbunden, aber liefert kein Bild.\n"
                                         "Ggf. Berechtigung prüfen oder wechseln.")
            else:
                self.video_label.setText("Kamera aktiv.")

    def switch_camera(self):
        """Schaltet zur nächsten Kamera durch."""
        self.current_cam_index += 1
        # Wir testen Indizes 0 bis 3, danach zurück zu 0
        if self.current_cam_index > 4:
            self.current_cam_index = 0

        self.switch_cam_btn.setText(f"Kamera wechseln (Aktuell: {self.current_cam_index})")
        self.initialize_camera()

    def open_data_folder(self):
        """Öffnet den Ordner mit den Trainingsdaten im Explorer/Finder."""
        if not self.custom_source_dir.exists():
            self.custom_source_dir.mkdir(parents=True, exist_ok=True)

        # Pfad in eine URL konvertieren, damit QDesktopServices ihn versteht
        folder_url = QUrl.fromLocalFile(str(self.custom_source_dir.absolute()))
        QDesktopServices.openUrl(folder_url)



    def create_button(self, text, slot, color):
        btn = QPushButton(text)
        btn.setFixedWidth(160)
        btn.setStyleSheet(f"background-color: {color}; color: black;")
        btn.pressed.connect(slot)
        self.sidebar_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.buttons.append(btn)
        return btn

    def initialize_model_system(self):
        """Initialisiert Ordnerstrukturen und lädt das Modell."""
        # Sicherstellen, dass das Basisverzeichnis existiert
        if not self.std_weights_path.exists():
            print("Standard-Modell nicht gefunden. Initiale Einrichtung startet...")
            try:
                self.setup_std_model()
            except Exception as e:
                QMessageBox.critical(self, "Fehler", str(e))
                return

        print(f"Lade YOLO Modell auf {DEVICE.upper()}...")
        try:
            # Versuchen, das Custom-Modell zu laden, falls vorhanden, sonst Standard
            weights_to_load = self.std_weights_path
            self.model = YOLO(weights_to_load)
            # self.model.to(DEVICE) # Ultralytics handelt dies oft automatisch, aber sicher ist sicher beim Predict call
            self.model_ready = True
            print(f"Modell geladen: {weights_to_load}")
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")

        # Vorbereitung für Nachtraining (Daten kopieren)
        self.prepare_custom_data_folder()

    def show_temp_message(self, message, duration=3000):
        self.message_label.setText(f"<i>{message}</i>")
        QTimer.singleShot(duration, self.message_label.clear)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        # OpenCV BGR zu RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        # GUI Update
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaledToHeight(TARGET_HEIGHT, Qt.TransformationMode.SmoothTransformation))

        # Modell Vorhersage
        if self.model_ready:
            results = self.model(frame_rgb, verbose=False, device=DEVICE)
            new_pred = results[0].probs.data.cpu().numpy()

            # Exponentielle Glättung (Smoothing)
            if self.current_probs is None:
                self.current_probs = new_pred
            else:
                self.current_probs = (SMOOTHING_ALPHA * new_pred) + ((1 - SMOOTHING_ALPHA) * self.current_probs)

            class_names_dict = results[0].names

            # Labels aktualisieren
            # Hinweis: Wir gehen davon aus, dass die Reihenfolge der Klassen im Modell konstant ist.
            # Besser wäre ein Mapping via class_names_dict, aber für YOLO Classification meist sortiert.

            top_pred_idx = self.current_probs.argmax()

            # Wir mappen die Modell-Klassen-Indices auf unsere GUI-Labels
            # Achtung: Die Reihenfolge der 'CATEGORIES' Liste muss mit der alphabetischen Reihenfolge
            # der Ordner im Dataset übereinstimmen (YOLO Standard), sonst sind die Labels vertauscht.
            # Hier nehmen wir an: 0: Gelber_Sack, 1: Papiermuell, 2: Restmuell -> Alphabetisch passt das meistens nicht
            # Gelber_Sack (G), Papiermuell (P), Restmuell (R). G < P < R. Passt zufällig.

            sorted_names = sorted(class_names_dict.values())  # YOLO sortiert Klassen alphabetisch beim Training

            for i, cat_name in enumerate(sorted_names):
                if i < len(self.prediction_labels):
                    prob = self.current_probs[i]
                    lbl = self.prediction_labels[i]

                    # Text Update
                    lbl.setText(f"<u>{cat_name}:</u> <strong>{float(prob) * 100:.2f}%</strong>")

                    # Highlight Top-Treffer
                    if i == top_pred_idx:
                        lbl.setStyleSheet("background-color: aquamarine; border: 1px solid green;")
                    else:
                        lbl.setStyleSheet("")

    def save_training_image(self, folder_name):
        """Speichert das aktuelle Bild in den Trainingsordner."""
        ret, frame = self.capture.read()
        if not ret:
            return

        target_dir = self.custom_source_dir / folder_name

        # Automatische Dateinamen-Generierung
        filename = self.get_next_filename(target_dir)
        cv2.imwrite(str(filename), frame)

        # Schöne Anzeige für den User
        display_name = folder_name.replace("_", " ")
        print(f"Bild gespeichert: {filename}")
        self.show_temp_message(f"Bild für <strong>{display_name}</strong> gespeichert.")

    def get_next_filename(self, folder_path, extension=".jpg"):
        """Findet den nächsten freien Dateinamen (nummeriert)."""
        folder_path.mkdir(parents=True, exist_ok=True)
        max_num = 0
        for file in folder_path.glob(f"*{extension}"):
            if file.stem.isdigit():
                num = int(file.stem)
                if num > max_num:
                    max_num = num
        return folder_path / f"{max_num + 1}{extension}"

    def prepare_custom_data_folder(self):
        """Erstellt eine Kopie der Standard-Daten für das Nachtraining."""
        if self.custom_source_dir.exists():
            shutil.rmtree(self.custom_source_dir)

        # Kopiere Standard-Daten als Basis
        if self.std_source_dir.exists():
            shutil.copytree(self.std_source_dir, self.custom_source_dir)
        else:
            self.custom_source_dir.mkdir(parents=True, exist_ok=True)

        print(f"Trainingsdaten-Verzeichnis vorbereitet: {self.custom_source_dir}")

    def create_yolo_dataset_structure(self, source, destination):
        """Konvertiert einfache Ordnerstruktur in YOLO train/val Struktur."""
        if destination.exists():
            shutil.rmtree(destination)

        modes = ['train', 'val']
        # Holen aller Unterordner (Kategorien)
        categories = [d for d in source.iterdir() if d.is_dir()]

        for mode in modes:
            for cat_dir in categories:
                dest_dir = destination / mode / cat_dir.name
                dest_dir.mkdir(parents=True, exist_ok=True)

                # Bilder kopieren
                for img in cat_dir.glob("*"):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        shutil.copy2(img, dest_dir / img.name)

    def start_finetuning(self):
        """Startet den Thread für das Nachtrainieren."""
        if not any(self.custom_source_dir.iterdir()):
            self.show_temp_message("Keine Daten zum Trainieren gefunden!", 5000)
            return

        print("Starte Nachtraining...")
        self.message_label.setText("Modell wird nachtrainiert... Bitte warten.")
        self.train_btn.setEnabled(False)
        for bnt in self.buttons:
            bnt.setEnabled(False)
        self.pbar.setVisible(True)
        self.pbar.setValue(0)
        self.model_ready = False  # Keine Vorhersagen während des Trainings (optional, spart Ressourcen)

        # Datensatz erstellen
        self.create_yolo_dataset_structure(self.custom_source_dir, self.custom_dataset_dir)

        # Alten Trainer aufräumen falls vorhanden
        if self.model:
            del self.model

        # Thread starten
        self.trainer = TrainerThread(
            dataset_path=self.custom_dataset_dir,
            project_path=self.project_path,
            run_name=self.custom_run_name,
            epochs=EPOCHS_TO_TRAIN
        )
        self.trainer.progress_update.connect(self.pbar.setValue)
        self.trainer.training_finished.connect(self.on_finetuning_finished)
        self.trainer.start()

    def on_finetuning_finished(self):
        """Wird aufgerufen, wenn der Trainer-Thread fertig ist."""
        print("Nachtraining abgeschlossen.")
        self.show_temp_message("Nachtrainierung erfolgreich! Neues Modell aktiv.", 5000)

        # Modell neu laden
        self.model = YOLO(self.custom_weights_path)
        self.model_ready = True

        # GUI zurücksetzen
        self.pbar.setVisible(False)
        for bnt in self.buttons:
            bnt.setEnabled(True)

    def setup_std_model(self):
        """Erstellt das initiale Modell, falls noch keines existiert."""
        print("Erstelle Basis-Modell...")

        required_folders = ["Gelber_Sack", "Papiermuell", "Restmuell"]
        missing_data = False

        for folder in required_folders:
            path = self.std_source_dir / folder
            path.mkdir(parents=True, exist_ok=True)
            # Prüfen ob Bilder drin sind
            if not any(path.iterdir()):
                print(f"Warnung: Ordner leer: {path}")
                missing_data = True

        if missing_data:
            raise Exception(f"Fehler: Bitte lege Startbilder in {self.std_source_dir} ab.")

        self.create_yolo_dataset_structure(self.std_source_dir, self.std_dataset_dir)

        # Initiales Training
        model = YOLO(CLASSIFICATION_MODEL)
        model.train(
            data=str(self.std_dataset_dir),
            epochs=EPOCHS_TO_TRAIN,
            imgsz=640,
            project=str(self.project_path),
            name=self.std_run_name,
            exist_ok=True,
            device=DEVICE
        )

        # Gewichte sichern
        src = (Path(model.trainer.save_dir) / "weights" / "last.pt").absolute()
        dst = self.std_weights_path.absolute()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and src != dst:
            shutil.copy2(src, dst)

        print("Basis-Modell erstellt.")

    def closeEvent(self, event):
        self.capture.release()
        event.accept()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    app = QApplication(sys.argv)
    window = TrashClassificatorApp()
    window.show()
    sys.exit(app.exec())