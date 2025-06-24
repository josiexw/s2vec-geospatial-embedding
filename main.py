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


# === Config ===
print("=== Configs ===")
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

NUM_EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Upload data ===
INPUT_DIR = 'uploads'
OUTPUT_DIR = 'patches'
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
input_path = os.path.join(INPUT_DIR)
output_path = os.path.join(OUTPUT_DIR)

all_gdfs = []
print("=== Uploading gdfs ===")
for file_name in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, file_name)

    # Process .gpkg (has layers)
    if path.endswith(".gpkg"):
        layers = pyogrio.list_layers(path)
        for layer in layers:
            layer_name = layer[0]
            print(f"Loading {layer_name} from {file_name}")
            try:
                gdf = gpd.read_file(path, layer=layer_name)
                if gdf.crs != "EPSG:3857":
                    gdf = gdf.to_crs("EPSG:3857")
                all_gdfs.append(gdf)
            except Exception as e:
                print(f"-- Skipping {layer_name}: {e}")
                continue
    else:
        print(f"Loading {file_name}")
        gdf = gpd.read_file(path)
        if gdf.crs != "EPSG:3857":
            gdf = gdf.to_crs("EPSG:3857")
        all_gdfs.append(gdf)

combined_gdf_wm = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
combined_gdf = combined_gdf_wm.to_crs("EPSG:4326")
print("Finished uploading gdfs")


# == Get features ===
print("=== Getting features ===")
all_columns = set()
for gdf in all_gdfs:
    all_columns.update(gdf.columns)
FEATURE_LIST = sorted(list(all_columns))
print(FEATURE_LIST)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_LIST)}

# === Partition data in S2 cells ===
level = 12  # Resolution of cells (8-18)
visited = set()

# Region bounds
minx, miny, maxx, maxy = combined_gdf.total_bounds
ll_sw = s2.LatLng.from_degrees(min(miny, maxy), min(minx, maxx))
ll_ne = s2.LatLng.from_degrees(max(miny, maxy), max(minx, maxx))
print(f"minx: {minx}, miny: {miny}, maxx: {maxx}, maxy: {maxy}")
region = s2.LatLngRect.from_point_pair(ll_sw, ll_ne)


print("=== Partitioning data ===")
def collect(cell_id):
    if cell_id.level() == level:
        visited.add(cell_id.id())
    else:
        for child in cell_id.children():
            covering = s2.RegionCoverer().get_covering(region)
            if child.id() in [c.id() for c in covering]:
                collect(child)


root = s2.CellId.from_lat_lng(ll_sw).parent(level)
collect(root)


# === Compute cell features ===
print("=== Computing cell features ===")
centroids = combined_gdf_wm.geometry.centroid

# Convert centroids back to lat/lng for S2
centroids_latlng = centroids.to_crs("EPSG:4326")
lats = centroids_latlng.y.values
lngs = centroids_latlng.x.values

# Compute S2 cell ids
cell_ids = [
    s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, lng)).parent(level).id()
    for lat, lng in zip(lats, lngs)
]

# Numeric features
numeric_cols = combined_gdf[FEATURE_LIST].select_dtypes(include=[np.number]).columns.tolist()
numeric_features = combined_gdf[numeric_cols].fillna(0).copy()
numeric_features["cell_id"] = cell_ids
agg_df = numeric_features.groupby("cell_id").sum()
features = {cid: row.values for cid, row in agg_df.iterrows()}

# Dissolve geometry
combined_gdf["cell_id"] = cell_ids
geom_per_cell = combined_gdf.dissolve(by="cell_id").geometry

rows = []
for cid, geom in geom_per_cell.items():
    centroid = geom.centroid
    area = geom.area
    bounds = geom.bounds
    numeric_vec = agg_df.loc[cid].values
    row = {
        "cell_id": cid,
        **{col: numeric_vec[i] for i, col in enumerate(agg_df.columns)},
        "centroid_x": centroid.x,
        "centroid_y": centroid.y,
        "area": area,
        "bounds_minx": bounds[0],
        "bounds_miny": bounds[1],
        "bounds_maxx": bounds[2],
        "bounds_maxy": bounds[3],
        "geometry_wkt": geom.wkt
    }
    rows.append(row)


# === Export as CSV ===
df = pd.DataFrame(rows)
df.to_csv("cell_features.csv", index=False)


# === Rasterize cell features into spatial grid images centered on each cell ===
W = 3
F = len(next(iter(features.values())))
print("=== Rasterizing data ===")
print("Feature vector length:", F)

def get_neighbor_ids(cell_id, window):
    neighbors = [cell_id]
    current = [cell_id]
    
    for _ in range(window):
        next_layer = []
        for cid in current:
            next_layer.extend(cid.get_edge_neighbors())
        neighbors.extend(next_layer)
        current = next_layer
        
    unique_neighbors = list({n.id(): n for n in neighbors}.values())
    return unique_neighbors

def rasterize(cell_id):
    feat_mat = np.zeros((W, W, F), dtype=np.float32)
    neighbors = get_neighbor_ids(cell_id, W)
    for i, n in enumerate(neighbors[:W * W]):
        feat_mat[i // W, i % W] = features.get(n.id(), np.zeros(F))
    return feat_mat

# Save patches
for cid in visited:
    patch = rasterize(s2.CellId(cid))
    patch_path = os.path.join(OUTPUT_DIR, f"{cid}.npy")
    np.save(patch_path, patch)


# === Pretrain ViT-style MAE ===
class SpatialPatchDataset(Dataset):
    def __init__(self, patch_paths):
        self.paths = patch_paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        patch = np.load(self.paths[i])
        return torch.tensor(patch).permute(2, 0, 1)  # (F, W, W)
    
class MaskedAutoencoder(torch.nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, num_heads, decoder_dim):
        super().__init__()
        self.model = MaskedAutoencoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_dim
        )

    def forward(self, x):
        return self.model(x)

    @property
    def encoder(self):
        return self.model.encoder

patch_files = sorted(glob.glob(f"{OUTPUT_DIR}/*.npy"))
dataset = SpatialPatchDataset(patch_files)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)
print("=== Training MAE ===")

model = MaskedAutoencoder(
    img_size=W, patch_size=1, in_chans=F,
    embed_dim=128, depth=6, num_heads=4,
    decoder_dim=128
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
