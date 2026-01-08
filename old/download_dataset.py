import os

# --- 1. SET CREDENTIALS BEFORE IMPORTING KAGGLE ---
# This tricks the library into thinking the environment is already set up.
os.environ['KAGGLE_USERNAME'] = "jonathanpi27"
os.environ['KAGGLE_KEY'] = "KGAT_f165dd17569ca42ed64b93b273da648b"

import shutil
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# --- CONFIGURATION ---
# DATASET_NAME = "mostafaabla/garbage-classification"
DATASET_NAME = "tommasosbrenna/recyclable-and-household-waste-white-background"
# ZIP_FILE = "garbage-classification.zip"
ZIP_FILE = "recyclable-and-household-waste-white-background.zip"
# RAW_DIR = "raw_download"
RAW_DIR = "raw_download_white"
# TARGET_DIR = "garbage_dataset/train"
TARGET_DIR = "garbage_dataset_white/train"

MAPPING = {
    "paper": "Papiermuell",
    "cardboard": "Papiermuell",
    "plastic": "Gelber_Sack",
    "metal": "Gelber_Sack",
    "biological": "Biomuell",
    "trash": "Restmuell",
    "battery": "Restmuell",
    "shoes": "Restmuell",
    "clothes": "Restmuell",
    "glass": "Restmuell",
    "brown-glass": "Restmuell",
    "green-glass": "Restmuell",
    "white-glass": "Restmuell"
}
MAPPING_WHITE = {
    "glass": "Altglas",
    "metal": "Gelber_Sack",
    "organic": "Biomuell",
    "paper": "Papiermuell",
    "plastic": "Gelber_Sack",
    "polystyrene": "Gelber_Sack",
    "textile": "Restmuell",
}


def setup_dataset():
    # 2. AUTHENTICATE
    # Now we can safely initialize the API
    api = KaggleApi()
    api.authenticate()

    # 3. DOWNLOAD
    if not os.path.exists(ZIP_FILE):
        print(f"Downloading {DATASET_NAME}...")
        try:
            api.dataset_download_files(DATASET_NAME, unzip=False, path=".", quiet=False)
            print("Download complete.")
        except Exception as e:
            print(f"Error: {e}")
            return
    else:
        print("Zip file already exists.")

    # 4. EXTRACT
    if not os.path.exists(RAW_DIR):
        print("Extracting files...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)

    # 5. ORGANIZE
    print("Organizing files...")

    # Find the internal root folder
    base_path = RAW_DIR
    for root, dirs, files in os.walk(RAW_DIR):
        if "plastic" in dirs and "paper" in dirs:
            base_path = root
            break

    count = 0
    for source_cat, target_cat in MAPPING.items():
        source_folder = os.path.join(base_path, source_cat)
        target_folder = os.path.join(TARGET_DIR, target_cat)

        os.makedirs(target_folder, exist_ok=True)

        if os.path.exists(source_folder):
            files = os.listdir(source_folder)
            for f in files:
                shutil.move(
                    os.path.join(source_folder, f),
                    os.path.join(target_folder, f)
                )
                count += 1

    print(f"Success! Moved {count} images.")


if __name__ == "__main__":
    setup_dataset()