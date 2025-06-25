import geopandas as gpd
import s2sphere as s2  # Python implementation of s2geometry
import pyogrio
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from third_party.mae.models_mae import MaskedAutoencoderViT
import glob
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os
import gc


# === Config ===
print("=== Configs ===")
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

NUM_EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if path.endswith(".gpkg"):
        layers = pyogrio.list_layers(path)
        for layer_name, _ in layers:
            try:
                gdf = gpd.read_file(path, layer=layer_name)
            except:
                continue
            if gdf.empty:
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

    else:
        try:
            gdf = gpd.read_file(path)
        except:
            continue
        if gdf.empty:
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


# === Pretrain ViT-style MAE ===
class SpatialPatchDataset(Dataset):
    def __init__(self, patch_paths):
        self.paths = patch_paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        patch = np.load(self.paths[i])
        return torch.tensor(patch).permute(2, 0, 1)  # (F, W, W)

patch_files = sorted(glob.glob(f"{PATCHES_DIR}/*.npy"))
dataset = SpatialPatchDataset(patch_files)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)
print("=== Training MAE ===")

# Infer F
sample_patch = np.load(patch_files[0])
F = sample_patch.shape[-1]

model = MaskedAutoencoderViT(
    img_size=W, patch_size=1, in_chans=F,
    embed_dim=128, depth=6, num_heads=4,
    decoder_embed_dim=128
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(NUM_EPOCHS):
    for imgs in loader:
        imgs = imgs.to(device)
        loss, _, _ = model(imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.encoder.state_dict(), "mae_encoder.pth")
print("Training complete")


# # === Finetuning ===
# encoder = model.encoder
# predictor = torch.nn.Linear(128, 1)
# criterion = torch.nn.MSELoss()
# opt = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-4)

# for imgs, labels in finetune_loader:
#     emb = encoder(imgs.to(device)).mean(dim=1)
#     preds = predictor(emb)
#     loss = criterion(preds.squeeze(), labels.to(device))
#     loss.backward(); opt.step(); opt.zero_grad()


# === Embed to text ===
def embedding_to_text(emb: np.ndarray, meta: dict):
    prompt = f"""
    You are a GIS expert. Given this spatial embedding and metadata:

    Embedding: {emb.tolist()}
    Metadata: {meta}

    Generate a human-friendly description, summarizing key features.
    """
    response = client.responses.create(
        model="gpt-4",
        input=prompt    
    )
    return response.output[0].content[0].text


# == Test ===
with torch.no_grad():
    first_batch = next(iter(loader)).to(device)
    emb = model.encoder(first_batch)
    emb_np = emb.cpu().numpy()[0]

print("Embedding:", emb.tolist())
meta = {"cell_id": 1234567890, "region": "Lausanne"}
description = embedding_to_text(emb_np, meta)
print(description)
