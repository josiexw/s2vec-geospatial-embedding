import geopandas as gpd
import s2sphere as s2  # Python implementation of s2geometry
import pyogrio
import pandas as pd
import numpy as np
import os
import gc


# === Config ===
print("=== Configs ===")
INPUT_DIR = 'uploads'
PATCHES_DIR = 'patches'
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(PATCHES_DIR, exist_ok=True)

level = 12  # S2 cell level
W = 3  # Window size
visited = set()
features = {}


# === Get features ===
print("=== Getting features ===")
feature_set = set()

for file_name in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, file_name)
    try:
        if path.endswith(".gpkg"):
            layers = pyogrio.list_layers(path)
            for layer in layers:
                gdf = gpd.read_file(path, layer=layer[0])
                feature_set.update(gdf.columns)
                del gdf; gc.collect()
        else:
            gdf = gpd.read_file(path)
            feature_set.update(gdf.columns)
            del gdf; gc.collect()
    except Exception as e:
        print(f"-- Skipping {file_name}: {e}")

FEATURE_LIST = sorted(list(feature_set))
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_LIST)}
print(f"Final feature list ({len(FEATURE_LIST)} features):", FEATURE_LIST)


# === Partition and rasterize S2 cells ===
print("=== Paritioning and rasterizing ===")

def encode_features(gdf, feature_list):
    encoded_cols = []
    col_names = []

    for col in feature_list:
        if col not in gdf.columns:
            encoded_cols.append(np.zeros((len(gdf), 1)))
            col_names.append(col)
            continue

        dtype = gdf[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            arr = gdf[col].fillna(0).values.reshape(-1, 1)
            if dtype == 'bool':
                arr = arr.astype(int)
            arr = (arr - arr.mean()) / (arr.std() + 1e-6)  # Normalize
            encoded_cols.append(arr)
            col_names.append(col)

        elif pd.api.types.is_object_dtype(dtype):
            unique_vals = gdf[col].dropna().unique()
            if len(unique_vals) <= 20:
                dummies = pd.get_dummies(gdf[col], prefix=col).astype(int)
                encoded_cols.append(dummies.values)
                col_names.extend(dummies.columns.tolist())
            else:
                lengths = gdf[col].fillna('').apply(lambda x: len(str(x).split())).values.reshape(-1, 1)
                encoded_cols.append(lengths)
                col_names.append(col + '_length')

        else:
            encoded_cols.append(np.zeros((len(gdf), 1)))
            col_names.append(col)

    full_encoded = np.hstack(encoded_cols)
    return pd.DataFrame(full_encoded, columns=col_names)

def get_neighbor_ids(cell_id, window):
    neighbors = [cell_id]
    current = [cell_id]
    for _ in range(window):
        next_layer = []
        for cid in current:
            next_layer.extend(cid.get_edge_neighbors())
        neighbors.extend(next_layer)
        current = next_layer
    return list({n.id(): n for n in neighbors}.values())

def rasterize_from_features(cell_id, local_features, feature_dim):
    feat_mat = np.zeros((W, W, feature_dim), dtype=np.float32)
    neighbors = get_neighbor_ids(cell_id, W)
    for i, n in enumerate(neighbors[:W * W]):
        feat_mat[i // W, i % W] = local_features.get(n.id(), np.zeros(feature_dim))
    return feat_mat

for file_name in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, file_name)

    if path.endswith(".gpkg") or path.endswith(".json"):
        layers = pyogrio.list_layers(path)
        for layer_name, _ in layers:
            try:
                gdf = gpd.read_file(path, layer=layer_name)
            except:
                continue
            if gdf.empty:
                print(f"Skipping empty layer: {layer_name}")
                continue
            if not isinstance(gdf, gpd.GeoDataFrame) or gdf.geometry.name not in gdf.columns:
                print(f"Skipping non-spatial layer: {layer_name}")
                continue
            if gdf.crs is None:
                print(f"Skipping layer with undefined CRS: {layer_name}")
                continue

            gdf = gdf.to_crs("EPSG:3857") if gdf.crs != "EPSG:3857" else gdf
            gdf["centroid"] = gdf.geometry.centroid
            gdf = gdf[gdf["centroid"].is_valid & ~gdf["centroid"].is_empty]
            gdf_centroids = gdf["centroid"].to_crs("EPSG:4326")

            # Drop nans
            gdf_centroids = gdf_centroids[~gdf_centroids.x.isna() & ~gdf_centroids.y.isna()]
            gdf_centroids = gdf_centroids[np.isfinite(gdf_centroids.x) & np.isfinite(gdf_centroids.y)]

            cell_ids = [
                s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, lng)).parent(level).id()
                for lat, lng in zip(gdf_centroids.y, gdf_centroids.x)
            ]

            encoded = encode_features(gdf, FEATURE_LIST)
            encoded["cell_id"] = cell_ids
            agg_df = encoded.groupby("cell_id").sum()
            local_features = {
                cid: row.values.astype(np.float32)
                for cid, row in agg_df.iterrows()
            }
            for cid in local_features:
                patch = rasterize_from_features(s2.CellId(cid), local_features, local_features[cid].shape[0])
                np.save(os.path.join(PATCHES_DIR, f"{cid}.npy"), patch)

    else:
        try:
            gdf = gpd.read_file(path)
        except:
            continue

        if gdf.empty:
                print(f"Skipping empty layer: {layer_name}")
                continue
        if not isinstance(gdf, gpd.GeoDataFrame) or gdf.geometry.name not in gdf.columns:
            print(f"Skipping non-spatial layer: {layer_name}")
            continue
        if gdf.crs is None:
            print(f"Skipping layer with undefined CRS: {layer_name}")
            continue

        gdf = gdf.to_crs("EPSG:3857") if gdf.crs != "EPSG:3857" else gdf
        gdf_centroids = gdf.geometry.centroid.to_crs("EPSG:4326")
        cell_ids = [
            s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, lng)).parent(level).id()
            for lat, lng in zip(gdf_centroids.y, gdf_centroids.x)
        ]

        encoded = encode_features(gdf, FEATURE_LIST)
        encoded["cell_id"] = cell_ids
        agg_df = encoded.groupby("cell_id").sum()
        local_features = {
            cid: row.values.astype(np.float32)
            for cid, row in agg_df.iterrows()
        }
        for cid in local_features:
            patch = rasterize_from_features(s2.CellId(cid), local_features, local_features[cid].shape[0])
            np.save(os.path.join(PATCHES_DIR, f"{cid}.npy"), patch)

print("Saved patches")
