import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

# === CONFIG ===
BATCH_SIZE = 32
EMBED_DIM = 512
TEXT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# === TEXT ENCODER ===
class TextEncoder(nn.Module):
    def __init__(self, model_name=TEXT_MODEL):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.model.config.hidden_size, EMBED_DIM)

    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        outputs = self.model(**tokens)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.projection(pooled)

# === IMAGE ENCODER (placeholder) ===
class DummyPatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*256, EMBED_DIM)  # Assume 256x256 grayscale images
        )

    def forward(self, x):
        return self.encoder(x)

# === CONTRASTIVE LOSS ===
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, image_embeds, text_embeds):
        image_embeds = nn.functional.normalize(image_embeds, dim=1)
        text_embeds = nn.functional.normalize(text_embeds, dim=1)
        logits = image_embeds @ text_embeds.T / self.temperature
        labels = torch.arange(len(image_embeds)).to(logits.device)
        loss_i2t = self.cross_entropy(logits, labels)
        loss_t2i = self.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2

# === DATASET ===
class PatchTextDataset(Dataset):
    def __init__(self, csv_path, image_folder):
        self.df = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_folder, f"{row['filename']}_{row['cell_id']}.png")
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        text = row['metadata_text']
        return image, text

# === TRAINING ===
def train():
    dataset = PatchTextDataset('./data/patch_metadata_pairs.csv', './data/patch_images')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    image_encoder = DummyPatchEncoder().cuda()
    text_encoder = TextEncoder().cuda()
    contrastive_loss = ContrastiveLoss()

    optimizer = torch.optim.AdamW(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4)

    for epoch in range(10):
        image_encoder.train()
        text_encoder.train()
        total_loss = 0

        for images, texts in dataloader:
            images = images.cuda()
            text_embeds = text_encoder(texts)
            image_embeds = image_encoder(images)
            loss = contrastive_loss(image_embeds, text_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

if __name__ == '__main__':
    train()
    