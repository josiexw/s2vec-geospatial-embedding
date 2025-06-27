import os
import zipfile
import shutil
import geopandas as gpd
import pyogrio
from deep_translator import GoogleTranslator

# === Setup ===
CURRENT_DIR = os.path.abspath("./gpkg_files")
TRANSLATOR = GoogleTranslator(source='auto', target='en')

def extract_gpkg(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            tmp_dir = zip_path + "_tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            zip_ref.extractall(tmp_dir)
            skip_move = False

            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    if file.endswith(".gpkg"):
                        target_path = os.path.join(CURRENT_DIR, file)
                        if os.path.exists(target_path):
                            skip_move = True
                            print(f"Skipping move for {file} — already exists.")
                            break

            if not skip_move:
                for root, _, files in os.walk(tmp_dir):
                    for file in files:
                        if file.endswith(".gpkg"):
                            shutil.move(os.path.join(root, file), os.path.join(CURRENT_DIR, file))

            shutil.rmtree(tmp_dir)
            os.remove(zip_path)
    except Exception as e:
        print(f"Failed to extract {zip_path}: {e}")

def translate_text(text):
    try:
        if isinstance(text, str) and text.strip():
            return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        pass
    return text

def translate_gpkg(filepath):
    try:
        layers = pyogrio.list_layers(filepath)
        EN_DIR = os.path.join(CURRENT_DIR, "translated")
        os.makedirs(EN_DIR, exist_ok=True)
        file_out = os.path.basename(filepath).replace(".gpkg", "_en.gpkg")
        out_path = os.path.join(EN_DIR, file_out)

        for layer in layers:
            try:
                gdf = gpd.read_file(filepath, layer=layer)
                if gdf.empty or gdf.geometry.is_empty.all():
                    print(f"Skipping layer '{layer}' — empty or no valid geometry.")
                    continue
            except Exception as e:
                print(f"Not translating unreadable layer '{layer}' in {filepath}: {e}")
                continue

            # Translate column names
            rename_map = {
                col: translate_text(col)
                for col in gdf.columns
                if col.lower() != 'geometry'
            }
            gdf = gdf.rename(columns=rename_map)

            # Translate string values
            for col in gdf.select_dtypes(include='object').columns:
                gdf[col] = gdf[col].apply(translate_text)

            try:
                gdf.to_file(out_path, driver="GPKG", layer=layer)
                print(f"== Wrote layer '{layer}' to {out_path}")
            except Exception as e:
                print(f"xx Failed to write layer '{layer}': {e}")

        print(f"Saved translated GPKG to: {out_path}")
    except Exception as e:
        print(f"Failed to translate {filepath}: {e}")

for file in os.listdir(CURRENT_DIR):
    full_path = os.path.join(CURRENT_DIR, file)
    if file.endswith(".zip"):
        extract_gpkg(full_path)
    elif file.endswith(".gpkg"):
        translate_gpkg(full_path)
