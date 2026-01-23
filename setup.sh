#!/bin/bash

# Funktion zum Prüfen ob ein Befehl existiert
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "[INFO] Prüfe System..."

PYTHON_CMD=""

# Suche nach Python 3.13
if command_exists python3.13; then
    PYTHON_CMD="python3.13"
elif command_exists python3; then
    # Prüfe Version
    VER=$(python3 -c"import sys; print(sys.version_info.minor)")
    if [ "$VER" -eq "13" ]; then
        PYTHON_CMD="python3"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "[WARNUNG] Python 3.13 nicht gefunden."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "[INFO] macOS erkannt. Versuche Installation via Homebrew..."
        if command_exists brew; then
            brew install python@3.13
            PYTHON_CMD="python3.13"
        else
            echo "[FEHLER] Homebrew nicht gefunden. Bitte installiere Python 3.13 manuell."
            exit 1
        fi
    elif command_exists apt-get; then
        echo "[INFO] Debian/Ubuntu erkannt. Füge deadsnakes PPA hinzu..."
        sudo apt-get update
        sudo apt-get install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update
        sudo apt-get install -y python3.13 python3.13-venv python3.13-dev
        PYTHON_CMD="python3.13"
    else
        echo "[FEHLER] Konnte Paketmanager nicht bestimmen. Bitte installiere Python 3.13 manuell."
        exit 1
    fi
fi

echo "[INFO] Nutze $PYTHON_CMD"

# Venv erstellen
if [ ! -d "venv" ]; then
    echo "[INFO] Erstelle Virtual Environment..."
    $PYTHON_CMD -m venv venv
else
    echo "[INFO] Virtual Environment existiert bereits."
fi

# Aktivieren und Installieren
source venv/bin/activate
echo "[INFO] Installiere Abhängigkeiten..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "[ERFOLG] Einrichtung abgeschlossen."
echo "Starte die App mit: python app.py"