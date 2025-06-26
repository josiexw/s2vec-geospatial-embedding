from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from third_party.mae.models_mae import MaskedAutoencoderViT
from openai import OpenAI
from dotenv import load_dotenv
import torch
import torch.nn as nn
import numpy as np
import os
import glob

# === Setup ===
print("=== Setting up Qdrant and OpenAI ===")
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

qdrant = QdrantClient(path="qdrant_db")
COLLECTION_NAME = "geo_embeddings"
VECTOR_DIM = 512

# === Create Collection ===
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

# === Load model and patch files ===
PATCHES_DIR = "patches"
patch_files = sorted(glob.glob(f"{PATCHES_DIR}/*.npy"))
F = max(np.load(p).shape[-1] for p in patch_files)
W = 3

model = MaskedAutoencoderViT(
    img_size=W, patch_size=1, in_chans=F,
    embed_dim=512,
    depth=12,
    num_heads=8,
    decoder_embed_dim=256
)
model.load_state_dict(torch.load("mae_model.pth", map_location="cpu"))
model.eval()

# === Embed and Upload ===
# print("=== Embedding and uploading to Qdrant ===")
# def pad_patch(patch, target_channels):
#     C, H, W = patch.shape
#     if C == target_channels:
#         return patch
#     padded = np.zeros((target_channels, H, W), dtype=np.float32)
#     padded[:C] = patch
#     return padded

# points = []
# for i, path in enumerate(patch_files):
#     patch = np.load(path).transpose(2, 0, 1)  # (F, H, W)
#     patch = pad_patch(patch, F)
#     tensor = torch.tensor(patch).unsqueeze(0)  # (1, F, H, W)

#     with torch.no_grad():
#         latent, _, _ = model.forward_encoder(tensor, mask_ratio=0.0)
#         emb = latent[0].mean(dim=0).cpu().numpy()

#     payload = {"file": os.path.basename(path)}
#     points.append(PointStruct(id=i, vector=emb.tolist(), payload=payload))

# qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
# print(f"Uploaded {len(points)} vectors to Qdrant.")

# === Retrieval Function ===
def retrieve_description(query_text: str, top_k=3):
    query_emb = client.embeddings.create(
        input=query_text,
        model="text-embedding-3-small"
    ).data[0].embedding

    reducer = nn.Linear(1536, 512)
    query_tensor = torch.tensor(query_emb).float().unsqueeze(0)
    query_proj = reducer(query_tensor).squeeze(0).detach().numpy()
    results = qdrant.query_points(collection_name=COLLECTION_NAME, query=query_proj, limit=top_k, with_vectors=True)

    context_blocks = []
    for i, point in enumerate(results.points):
        emb = point.vector
        payload = point.payload
        block = f"Result {i+1}:\nEmbedding: {emb[:10]}... (truncated)\nMetadata: {payload}\n"
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    prompt = f"""
    You are a GIS analyst. A user asked the following question:
    "{query_text}"
    You have access to the following spatial embeddings and metadata from similar regions:
    {context}
    Based on this, provide a helpful, human-friendly answer to the user question using information from the results.
    Also summarize the data you are provided.
    """.strip()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# === Example Query ===
query = "How many neighborhoods are in Biel?"
response = retrieve_description(query)
print(f"--- Response ---\n{response}\n")
