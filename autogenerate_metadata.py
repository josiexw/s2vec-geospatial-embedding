import geopandas as gpd
import pyogrio
import pandas as pd
import os
import s2sphere as s2
from openai import OpenAI
from dotenv import load_dotenv

# === Config ===
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

INPUT_DIR = 'uploads'
METADATA_CSV = 'patch_metadata.csv'
NL_METADATA_CSV = 'patch_nl_metadata.csv'
level = 12

rows = []
nl_rows = []

# === Get metadata ===
def get_metadata_rows(gdf, file_name, layer_name):
    gdf = gdf.to_crs("EPSG:3857") if gdf.crs != "EPSG:3857" else gdf
    gdf["centroid"] = gdf.geometry.centroid
    gdf = gdf[gdf["centroid"].is_valid & ~gdf["centroid"].is_empty]
    gdf_centroids = gdf["centroid"].to_crs("EPSG:4326")

    cell_ids = [
        s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, lng)).parent(level).id()
        for lat, lng in zip(gdf_centroids.y, gdf_centroids.x)
    ]
    gdf["cell_id"] = cell_ids
    for cid, group in gdf.groupby("cell_id"):
        coords = group["centroid"].apply(lambda pt: f"({pt.y:.5f}, {pt.x:.5f})").tolist()
        features = list(set(group.columns) - {"geometry", "centroid", "cell_id"})
        feature_data = group[features].fillna("").astype(str).apply(lambda row: row.to_dict(), axis=1).tolist()
        rows.append({
            "filename": file_name,
            "layer": layer_name,
            "cell_id": cid,
            "features": ", ".join(features),
            "coordinates": str(coords),
            "feature_info": str(feature_data)
        })

        prompt = f"""
        Given the following geospatial patch metadata, generate 20 human-friendly descriptions.
        Each description should mention the features and coordinates, be phrased differently, and have maximum 50 words.
        Separate each description with ===.

        Features: {features}
        Coordinates: {coords}
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            descriptions = response["choices"][0]["message"]["content"].split("===")
            for desc in descriptions:
                clean = desc.strip()
                if clean:
                    nl_rows.append({
                        "filename": file_name,
                        "layer": layer_name,
                        "cell_id": cid,
                        "nl_metadata": clean
                    })
        except Exception as e:
            print(f"OpenAI API failed on patch {cid}: {e}")

# === Process files ===
for file_name in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, file_name)
    try:
        if path.endswith(".gpkg"):
            layers = pyogrio.list_layers(path)
            for layer in layers:
                try:
                    gdf = gpd.read_file(path, layer=layer[0])
                    if gdf.empty or gdf.crs is None:
                        continue
                    get_metadata_rows(gdf, file_name, layer[0])
                except Exception as e:
                    print(f"Skipping layer {layer[0]} in {file_name}: {e}")
        else:
            gdf = gpd.read_file(path)
            if gdf.empty or gdf.crs is None:
                continue
            layer_name = os.path.splitext(file_name)[0]
            get_metadata_rows(gdf, file_name, layer_name)
    except Exception as e:
        print(f"Skipping file {file_name}: {e}")

# === Save to CSV ===
df = pd.DataFrame(rows)
df.to_csv(METADATA_CSV, index=False)
print(f"Saved metadata to {METADATA_CSV}")

nl_df = pd.DataFrame(nl_rows)
nl_df.to_csv(NL_METADATA_CSV, index=False)
print(f"Saved natural language metadata to {NL_METADATA_CSV}")
