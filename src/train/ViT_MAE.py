import torch
from torch.utils.data import Dataset, DataLoader
from third_party.mae.models_mae import MaskedAutoencoderViT
import glob
import numpy as np
import os


# === Config ===
print("=== Configs ===")
PATCHES_DIR = 'patches'
NUM_EPOCHS = 10
W = 3  # Window size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Pretrain ViT-style MAE ===
def pad_patch(patch, target_channels):
    C, H, W = patch.shape
    if C == target_channels:
        return patch
    padded = np.zeros((target_channels, H, W), dtype=np.float32)
    padded[:C] = patch
    return padded

class SpatialPatchDataset(Dataset):
    def __init__(self, patch_paths, target_channels):
        self.paths = patch_paths
        self.target_channels = target_channels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        patch = np.load(self.paths[i])
        patch = pad_patch(patch.transpose(2, 0, 1), self.target_channels)  # (F, H, W)
        patch = torch.tensor(patch)
        return patch

patch_files = sorted(glob.glob(f"{PATCHES_DIR}/*.npy"))
max_channels = max(np.load(p).shape[-1] for p in patch_files)
F = max_channels
dataset = SpatialPatchDataset(patch_files, max_channels)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)

print("=== Training MAE ===")
model = MaskedAutoencoderViT(
    img_size=W, patch_size=1, in_chans=F,
    embed_dim=512,
    depth=12,
    num_heads=8,
    decoder_embed_dim=256
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(NUM_EPOCHS):
    print(f"=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
    for batch_idx, imgs in enumerate(loader):
        imgs = imgs.to(device)
        loss, _, _ = model(imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(loader)}] Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "mae_model.pth")
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
