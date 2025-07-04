import geopandas as gpd
import pyogrio
import pandas as pd
import os
import s2sphere as s2
from functools import lru_cache
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# === Config ===
INPUT_DIR = 'gpkg_files'
METADATA_PARQUET = './data/metadata/patch_metadata'
LEVELS = [10, 12, 14, 16, 18]
BATCH_SIZE = 10
geolocator = Nominatim(user_agent="geo_nl_metadata", timeout=5, scheme='http', domain="localhost:8080")

@lru_cache(maxsize=10000)
def cached_reverse_geocode(coord):
    try:
        result = geolocator.reverse(coord, timeout=5)
        time.sleep(0.05)
        return result.address if result else ""
    except GeocoderTimedOut:
        print(f"Timeout getting address for {coord}, skipping.")
        return ""
    except Exception as e:
        print(f"Failed to get address for {coord}: {e}")
        return ""

seen_coords = {}
def batch_reverse_geocode(coords):
    addresses = []
    for i in range(0, len(coords), BATCH_SIZE):
        batch = coords[i:i + BATCH_SIZE]
        batch_addresses = []
        for coord in batch:
            if coord in seen_coords:
                batch_addresses.append(seen_coords[coord])
            else:
                address = cached_reverse_geocode(coord)
                batch_addresses.append(address)
                seen_coords[coord] = address
        addresses.extend(batch_addresses)
    return addresses

def get_metadata_rows(gdf, file_name, layer_name):
    gdf_projected = gdf.to_crs("EPSG:3857")
    gdf["centroid"] = gdf_projected.geometry.centroid.to_crs("EPSG:4326")
    gdf = gdf.loc[gdf["centroid"].is_valid & ~gdf["centroid"].is_empty].copy()
    rows = []

    for level in LEVELS:
        gdf["cell_id"] = [
            s2.CellId.from_lat_lng(s2.LatLng.from_degrees(pt.y, pt.x)).parent(level).id()
            for pt in gdf["centroid"]
        ]

        for cid, group in gdf.groupby("cell_id"):
            coords = [(pt.y, pt.x) for pt in group["centroid"]]
            features = list(set(group.columns) - {"geometry", "centroid", "cell_id"})
            feature_data = group[features].fillna("").astype(str).apply(lambda row: row.to_dict(), axis=1)
            # addresses = batch_reverse_geocode(coords)
            addresses = []  # Removed addresses because it was taking too long, also 

            row = {
                "filename": file_name,
                "layer": layer_name,
                "level": level,
                "cell_id": cid,
                "features": "; ".join(features),
                "coordinates": "; ".join(f"({lat:.5f}, {lon:.5f})" for lat, lon in coords),
                "addresses": "; ".join(addresses),
                "feature_info": "; ".join(map(str, feature_data))
            }

            rows.append(row)
    return pd.DataFrame(rows)

def main():
    for file_name in os.listdir(INPUT_DIR):
        parquet_filename = f"{METADATA_PARQUET}_{file_name}.parquet"

        # Skip existing files
        if os.path.exists(parquet_filename):
            continue

        pd.DataFrame(columns=[
            "filename", "layer", "level", "cell_id",
            "features", "coordinates", "addresses", "feature_info"
        ]).to_parquet(parquet_filename, index=False)

        path = os.path.join(INPUT_DIR, file_name)
        try:
            all_rows = pd.DataFrame()
            layers = pyogrio.list_layers(path)
            for layer in layers:
                try:
                    gdf = gpd.read_file(path, layer=layer[0])
                    if gdf.empty or gdf.crs is None:
                        continue
                    gdf = gdf.to_crs("EPSG:4326")
                    rows = get_metadata_rows(gdf, file_name, layer[0])
                    all_rows = pd.concat([all_rows, rows], ignore_index=True)
                except Exception as e:
                    print(f"Skipping layer {layer[0]} in {file_name}: {e}")
            
            all_rows.to_parquet(parquet_filename, index=False)
            print(f"Finished: {file_name}")
        except Exception as e:
            print(f"Skipping file {file_name}: {e}")

if __name__ == "__main__":
    main()
