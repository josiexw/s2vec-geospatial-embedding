import json
import csv
import os
import torch
from langdetect import detect
import yake
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np

INPUT_PATH = "./data/opendataswiss.parquet.parquet"
OUTPUT_PATH = "./data/train.parquet"
CITY_LIST_PATH = "./data/swiss_cities.txt"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

mlm_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
mlm_model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-uncased")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlm_model.to(DEVICE)
mlm_model.eval()

with open("./data/query_templates.json", "r", encoding="utf-8") as f:
    TEMPLATES = json.load(f)

with open(CITY_LIST_PATH, "r", encoding="utf-8") as f:
    SWISS_CITIES = set(line.strip().lower() for line in f if line.strip())

EXTRACTORS = {}
def extract_keywords(text, top_k=5, lang='en'):
    try:
        if lang not in EXTRACTORS:
            EXTRACTORS[lang] = yake.KeywordExtractor(lan=lang, n=3, top=top_k)
        kws = EXTRACTORS[lang].extract_keywords(text)
        return [kw.lower().strip() for kw, _ in kws]
    except:
        return []

def contains_swiss_city(text):
    return next((word for word in text.lower().split() if word in SWISS_CITIES), None)

def generate_template_queries(keywords, city=None, lang="en"):
    queries = []
    for kw in keywords:
        for template in TEMPLATES.get(lang, []):
            q = template.format(kw=kw)
            if city and np.random.rand() < 0.5:
                q += f" in {city}"
            queries.append((q, lang))
    return queries

def mlm_score_batch(queries, prefix="the user queries for ", batch_size=16):
    scores = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        full_sentences = [f"{prefix}{q}" for q in batch]
        with torch.no_grad():
            inputs = mlm_tokenizer(full_sentences, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = mlm_model(**inputs, labels=inputs["input_ids"])
            losses = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                inputs["input_ids"].view(-1),
                reduction="none"
            )
            token_lens = inputs["attention_mask"].sum(dim=1)
            token_losses = losses.view(inputs["input_ids"].size()).sum(dim=1)
            per_example_loss = (token_losses / token_lens).cpu().tolist()
            scores.extend(per_example_loss)
    return scores

def semantic_filter(queries, max_loss=5.0):
    texts = [q for q, _ in queries]
    langs = [lang for _, lang in queries]
    losses = mlm_score_batch(texts)
    return [(q, lang, loss) for q, lang, loss in zip(texts, langs, losses) if loss < max_loss]

def load_existing_ids(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return set(row["id"] for row in reader if "id" in row)

def main():
    existing_ids = load_existing_ids(OUTPUT_PATH)
    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, open(OUTPUT_PATH, "a", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        if os.stat(OUTPUT_PATH).st_size == 0:
            writer.writerow(["query", "id", "language", "mlm_loss"])

        for i, line in enumerate(f_in):
            dataset = json.loads(line)
            dataset_id = dataset.get("id")
            if dataset_id in existing_ids:
                continue

            title = dataset.get("short_description", "")
            description = dataset.get("description", "")
            keywords = dataset.get("keywords", [])
            combined_text = f"{title} {description} {' '.join(keywords)}"

            try:
                dom_lang = detect(combined_text)
                extracted_keywords = extract_keywords(combined_text, top_k=5, lang=dom_lang)
                if not extracted_keywords:
                    continue
            except:
                continue

            city = contains_swiss_city(title)
            raw_queries = generate_template_queries(extracted_keywords, city, lang=dom_lang)
            if not raw_queries:
                continue

            valid_queries = semantic_filter(raw_queries)
            if not valid_queries:
                continue

            for query, lang, score in valid_queries:
                writer.writerow([query, dataset_id, lang, score])
            if i % 10 == 0:
                print(f"Finished querying dataset {i}: {dataset_id}")

if __name__ == "__main__":
    main()
