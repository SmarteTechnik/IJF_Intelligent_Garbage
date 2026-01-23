@echo off
setlocal

echo [INFO] Pruefe Python Installation...

REM Pruefe ob py launcher installiert ist und Python 3.13 verfuegbar ist
py -3.13 --version >nul 2>&1
if %errorlevel% NEQ 0 (
    echo [WARNUNG] Python 3.13 wurde nicht gefunden.
    echo [INFO] Versuche Installation via Winget...
    winget install -e --id Python.Python.3.13
    if %errorlevel% NEQ 0 (
        echo [FEHLER] Automatische Installation fehlgeschlagen.
        echo Bitte installiere Python 3.13 manuell von python.org.
        pause
        exit /b 1
    )
    echo [INFO] Bitte starte dieses Skript neu, nachdem die Installation abgeschlossen ist.
    pause
    exit /b 0
)

echo [INFO] Python 3.13 gefunden. Erstelle Virtual Environment...

if not exist venv (
    py -3.13 -m venv venv
    echo [INFO] Venv erstellt.
) else (
    echo [INFO] Venv existiert bereits.
)

echo [INFO] Aktiviere Venv und installiere Requirements...
call venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [ERFOLG] Installation abgeschlossen!
echo Starte die Anwendung mit: python app.py
echo.
pause