import geopandas as gpd
import s2sphere as s2
import pyogrio
import pandas as pd
import numpy as np
import os
import gc
import torch    
from tqdm import tqdm
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer

INPUT_DIR = 'gpkg_files'
PATCHES_DIR = 'patches'
PATCH_INFO_DIR = './data/patches_info/patch_embedding'
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(PATCHES_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

LEVELS = [10, 12, 14, 16, 18]
W = 3

def normalize_name(name):
    name = name.lower().replace("_", "").replace("-", "")
    return ''.join(filter(str.isalnum, name))

def auto_group_columns(columns, threshold=0.8):
    grouped = {}
    for col in columns:
        norm_col = normalize_name(col)
        match = get_close_matches(norm_col, grouped.keys(), n=1, cutoff=threshold)
        if match:
            grouped[match[0]].append(col)
        else:
            grouped[norm_col] = [col]
    return grouped

def encode_grouped_features(gdf, grouped_cols, use_bert=True):
    feature_vecs = {}
    for group_name, cols in grouped_cols.items():
        valid_cols = [col for col in cols if col in gdf.columns]
        if not valid_cols:
            continue

        sample = gdf[valid_cols[0]]

        if pd.api.types.is_numeric_dtype(sample):
            values = gdf[valid_cols].mean(axis=1)
            feature_vecs[group_name] = values

        elif pd.api.types.is_object_dtype(sample) and use_bert:
            if len(valid_cols) == 0 or gdf[valid_cols].dropna(how='all').empty:
                continue
            agg_result = gdf[valid_cols].astype(str).agg(lambda x: " ".join(set(x)), axis=1)
            if isinstance(agg_result, pd.DataFrame):
                agg_result = agg_result.iloc[:, 0]
            combined_text = agg_result.fillna("")
            batch_embeddings = bert_model.encode(combined_text.tolist(), convert_to_tensor=False, show_progress_bar=False)
            if len(batch_embeddings) == 0:
                continue
            bert_array = np.vstack(batch_embeddings)
            for i in range(bert_array.shape[1]):
                feature_vecs[f"{group_name}_bert_{i}"] = bert_array[:, i]
        else:
            counts = gdf[valid_cols].notnull().sum(axis=1)
            feature_vecs[group_name + "_nonnull"] = counts
    return pd.DataFrame(feature_vecs)

def get_neighbor_ids(cell_id, window):
    neighbors = {cell_id.id(): cell_id}
    current = [cell_id]
    for _ in range(window):
        next_layer = []
        for cid in current:
            for neighbor in cid.get_edge_neighbors():
                neighbors[neighbor.id()] = neighbor
                next_layer.append(neighbor)
        current = next_layer
    return list(neighbors.values())

def rasterize_from_features(cell_id, local_features, feature_dim):
    feat_mat = np.zeros((W, W, feature_dim), dtype=np.float32)
    neighbors = get_neighbor_ids(cell_id, W)
    for i, n in enumerate(neighbors[:W * W]):
        feat_mat[i // W, i % W] = local_features.get(n.id(), np.zeros(feature_dim))
    return feat_mat

print("=== Processing files ===")
for filename in tqdm(os.listdir(INPUT_DIR)):
    filename_short = filename.replace(".gpkg", "")
    path = os.path.join(INPUT_DIR, filename)
    parquet_filename = f"{PATCH_INFO_DIR}_{filename_short}.parquet"

    if os.path.exists(parquet_filename):
        continue

    try:
        layers = pyogrio.list_layers(path)
    except:
        continue

    file_records = []
    for layer_name, _ in layers:
        try:
            gdf = gpd.read_file(path, layer=layer_name)
        except:
            continue

        if gdf.empty or not isinstance(gdf, gpd.GeoDataFrame) or gdf.crs is None:
            continue

        gdf = gdf.to_crs("EPSG:3857")
        gdf["centroid"] = gdf.geometry.centroid
        gdf = gdf[gdf["centroid"].is_valid & ~gdf["centroid"].is_empty]

        gdf_centroids = gdf["centroid"].to_crs("EPSG:4326")
        valid_mask = gdf_centroids.x.notna() & gdf_centroids.y.notna()
        valid_mask &= np.isfinite(gdf_centroids.x) & np.isfinite(gdf_centroids.y)

        gdf_centroids = gdf_centroids[valid_mask]
        gdf = gdf[valid_mask]

        raw_cols = [col for col in gdf.columns if col not in ["geometry", "centroid"]]
        FEATURE_LIST = auto_group_columns(raw_cols)
        encoded = encode_grouped_features(gdf, FEATURE_LIST)

        for level in LEVELS:
            try:
                cell_ids = [
                    s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, lng)).parent(level).id()
                    for lat, lng in zip(gdf_centroids.y, gdf_centroids.x)
                ]
                encoded["cell_id"] = cell_ids
                agg_df = encoded.groupby("cell_id").sum()
                local_features = {
                    cid: row.values.astype(np.float32)
                    for cid, row in agg_df.iterrows()
                }
                for cid, embedding in local_features.items():
                    patch = rasterize_from_features(s2.CellId(cid), local_features, embedding.shape[0])
                    np.save(os.path.join(PATCHES_DIR, f"{filename_short}_{layer_name}_level{level}_{cid}.npy"), patch)
                    file_records.append({
                        "filename": filename,
                        "layer": layer_name,
                        "level": level,
                        "cell_id": cid,
                        "embedding": embedding.tolist()
                    })
                del agg_df, local_features
            except Exception as e:
                print(f"  -- Skipping level {level} in {filename}/{layer_name}: {e}")
            gc.collect()
        del gdf, encoded, gdf_centroids
        gc.collect()

    pd.DataFrame(file_records).to_parquet(parquet_filename, index=False)
    print(f"Saved: {parquet_filename}")
