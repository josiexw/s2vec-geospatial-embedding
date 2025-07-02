import os
import zipfile
import shutil
import geopandas as gpd
import pyogrio
from deep_translator import GoogleTranslator
from functools import lru_cache
from multiprocessing import Pool, cpu_count

# === Setup ===
CURRENT_DIR = os.path.abspath("./gpkg_files")
TRANSLATOR = GoogleTranslator(source='auto', target='en')

TRANSLATED_DIR = os.path.join(CURRENT_DIR, "translated")
os.makedirs(TRANSLATED_DIR, exist_ok=True)

@lru_cache(maxsize=None)
def fast_translate(text):
    try:
        if isinstance(text, str) and text.strip():
            return TRANSLATOR.translate(text)
    except Exception:
        pass
    return text

def translate_column(col):
    unique_vals = col.dropna().unique()
    translations = {val: fast_translate(val) for val in unique_vals}
    return col.map(translations).fillna(col)

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

def translate_gpkg(filepath):
    try:
        file_out = os.path.basename(filepath).replace(".gpkg", "_en.gpkg")
        out_path = os.path.join(TRANSLATED_DIR, file_out)
        if os.path.exists(out_path):
            print(f"Already translated: {out_path}")
            return

        layers = pyogrio.list_layers(filepath)

        for layer in layers:
            layer_name = layer[0]
            print(f"Translating {layer_name} in {filepath}")
            try:
                gdf = gpd.read_file(filepath, layer=layer_name)
                if gdf.empty or gdf.geometry.is_empty.all():
                    print(f"Skipping layer {layer_name}: empty or no valid geometry.")
                    continue
            except Exception as e:
                print(f"Not translating unreadable layer '{layer}' in {filepath}: {e}")
                continue

            # Translate column names
            rename_map = {
                col: fast_translate(col)
                for col in gdf.columns if col.lower() != 'geometry'
            }
            gdf = gdf.rename(columns=rename_map)

            # Translate string values
            for col in gdf.select_dtypes(include='object').columns:
                gdf[col] = translate_column(gdf[col])

            try:
                gdf.to_file(out_path, driver="GPKG", layer=layer_name)
                print(f"== Wrote layer '{layer_name}' to {out_path}")
            except Exception as e:
                print(f"xx Failed to write layer {layer_name}: {e}")

        print(f"Saved translated GPKG to: {out_path}")
    except Exception as e:
        print(f"Failed to translate {filepath}: {e}")

def process_file(file):
    full_path = os.path.join(CURRENT_DIR, file)
    if file.endswith(".zip"):
        extract_gpkg(full_path)
    elif file.endswith(".gpkg"):
        translate_gpkg(full_path)

if __name__ == "__main__":
    files = [f for f in os.listdir(CURRENT_DIR) if f.endswith(".zip") or f.endswith(".gpkg")]
    with Pool(processes=min(cpu_count(), 4)) as pool:  # limit
        pool.map(process_file, files)
