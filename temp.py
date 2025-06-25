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

INPUT_DIR = 'uploads'
PATCHES_DIR = 'patches'
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(PATCHES_DIR, exist_ok=True)

level = 12  # S2 cell level
W = 3  # Window size
visited = set()
features = {}


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
