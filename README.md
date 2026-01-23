# Intelligenter Müll-Trenner (Smart Trash Classifier)

Diese Anwendung nutzt Computer Vision und Deep Learning (YOLOv26), um Abfall über eine Webcam zu klassifizieren und hilft bei der korrekten Mülltrennung (Gelber Sack, Papiermüll, Restmüll).

Die App basiert auf PyQt6 für die Oberfläche und Ultralytics YOLO für die KI. Sie bietet eine Funktion zum **Nachtrainieren** (Fine-Tuning) direkt aus der Benutzeroberfläche heraus.

## 🚀 Features

* **Live-Erkennung:** Klassifiziert Objekte in Echtzeit über die Webcam.
* **Auto-Hardware-Beschleunigung:** Nutzt automatisch CUDA (NVIDIA), MPS (Mac Silicon) oder CPU.
* **Interaktives Nachtrainieren:**
    * Falsche Erkennung? Ein Klick speichert das Bild in der korrekten Kategorie.
    * Integrierter Trainingsprozess aktualisiert das KI-Modell im Hintergrund.
* **Deutsche Lokalisierung:** Vollständig übersetzte Oberfläche und Logs.

## 🛠 Voraussetzungen

* Python 3.13
* Webcam

## 📦 Installation

### Windows
1.  Doppelklicke auf `setup.bat`.
2.  Das Skript prüft, ob Python 3.13 installiert ist (und installiert es ggf. nach), erstellt eine virtuelle Umgebung und installiert alle Bibliotheken.

### macOS / Linux
1.  Öffne ein Terminal im Ordner.
2.  Mache das Skript ausführbar: `chmod +x setup.sh`
3.  Führe es aus: `./setup.sh`

## ▶️ Starten

Nach der Installation kannst du die App wie folgt starten:

**Windows:**
```cmd
venv\Scripts\python app.py
```


## Erstellen einer .exe datei für Windows

Nachdem das Setup mit der .bat Datei abgeschlossen ist, kann eine Exe Datei erstellt werden:


```pyinstaller --noconsole --onefile --icon="imgs\icon.ico" --name="TrashClassifier" app.py```


## Erstellen eines .dmg für macOS
Ohne das nutzen der setup.sh Script kann mit diesem Befehl eine .dmg Datei erstellt werden:

```./build_dmg.sh```

Es kann sein, dass die Anwendung bei ersten Start nicht funktioniert, 
da erst die Kamera-Berechtigung angefragt wird. Beim zweiten Start sollte es aber klappen. 