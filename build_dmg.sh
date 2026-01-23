#!/bin/bash

# ==============================================================================
#  macOS DMG Builder für TrashClassifier (Mit Kamera-Berechtigung)
# ==============================================================================

# --- Konfiguration ---
APP_NAME="TrashClassifier"
MAIN_SCRIPT="app.py"
DATA_FOLDER="DATA"
DMG_NAME="${APP_NAME}_Application.dmg"
VENV_DIR="venv_build"
SPEC_FILE="${APP_NAME}.spec"
ICON_FILE="imgs/icon.icns"

# Farben
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}[INFO] Starte Build-Prozess für $APP_NAME...${NC}"

# 1. System-Check
if ! command -v brew &> /dev/null; then
    echo -e "${RED}[FEHLER] Homebrew fehlt. Bitte installiere es: https://brew.sh/${NC}"
    exit 1
fi

if ! command -v create-dmg &> /dev/null; then
    echo -e "${BLUE}[INFO] Installiere 'create-dmg'...${NC}"
    brew install create-dmg
fi

# Check for Icon
ICON_CONFIG="None"
if [ -f "$ICON_FILE" ]; then
    echo -e "${GREEN}[OK] Icon file '$ICON_FILE' found.${NC}"
    ICON_CONFIG="'$ICON_FILE'"
else
    echo -e "${RED}[WARNING] '$ICON_FILE' not found! App will have generic python icon.${NC}"
fi

# 2. Virtual Environment
echo -e "${BLUE}[INFO] Erstelle Build-Umgebung...${NC}"
rm -rf $VENV_DIR
python3.13 -m venv $VENV_DIR || python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

echo -e "${BLUE}[INFO] Installiere Abhängigkeiten...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# 3. SPEC Datei erstellen (WICHTIG: Hier werden die Berechtigungen gesetzt)
echo -e "${BLUE}[INFO] Erstelle PyInstaller Spec-File mit Kamera-Rechten...${NC}"

# Wir schreiben die Spec-Datei direkt via Python-Code in eine Datei
cat <<EOF > "$SPEC_FILE"
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['$MAIN_SCRIPT'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['ultralytics', 'torch', 'torchvision', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='$APP_NAME',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',
)

app = BUNDLE(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='$APP_NAME.app',
    icon=$ICON_CONFIG,
    bundle_identifier='com.trashclassifier.app',
    info_plist={
        'NSCameraUsageDescription': 'Diese App benötigt Zugriff auf die Kamera, um Müll zu scannen und zu klassifizieren.',
        'CFBundleDisplayName': 'Müll Trenner',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': 'True'
    },
)
EOF

# 4. Build starten (basierend auf der Spec-Datei)
echo -e "${BLUE}[INFO] Baue Applikation...${NC}"
pyinstaller --clean --noconfirm "$SPEC_FILE"

if [ ! -d "dist/$APP_NAME.app" ]; then
    echo -e "${RED}[FEHLER] Build fehlgeschlagen.${NC}"
    exit 1
fi

# 5. DATA Ordner integrieren
echo -e "${BLUE}[INFO] Kopiere Daten...${NC}"
DEST_DIR="dist/$APP_NAME.app/Contents/MacOS"
cp -r "$DATA_FOLDER" "$DEST_DIR/"

# 6. Code Signing (Ad-Hoc - Nötig damit die Berechtigungen greifen)
echo -e "${BLUE}[INFO] Signiere die App (Ad-Hoc)...${NC}"
codesign --force --deep --sign - "dist/$APP_NAME.app"

# 7. DMG erstellen
echo -e "${BLUE}[INFO] Erstelle DMG...${NC}"
rm -f "$DMG_NAME"

create-dmg \
  --volname "${APP_NAME} Installer" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --hide-extension "${APP_NAME}.app" \
  --app-drop-link 600 185 \
  "$DMG_NAME" \
  "dist/$APP_NAME.app"

# Cleanup
deactivate
# rm -rf $VENV_DIR
# rm "$SPEC_FILE" # Optional: Spec file löschen

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}   FERTIG! DMG erstellt: $PWD/$DMG_NAME${NC}"
echo -e "${GREEN}============================================================${NC}"