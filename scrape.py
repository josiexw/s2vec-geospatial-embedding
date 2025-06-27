from turtle import down
import requests
import os
import csv
from deep_translator import GoogleTranslator
import gc

TRANSLATOR = GoogleTranslator(source='auto', target='en')
LANGUAGES = ['de', 'fr', 'it', 'en']
SAVE_DIR = "gpkg_files"
CSV_PATH = "geospatial_swiss.csv"
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024
os.makedirs(SAVE_DIR, exist_ok=True)

API_URL = "https://opendata.swiss/api/3/action/package_search"
params = {"q": "res_format:GPKG", "rows": 600}

def translate_text(text):
    try:
        if isinstance(text, str) and text.strip():
            return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        pass
    return text

response = requests.get(API_URL, params=params)
data = response.json()

for result in data["result"]["results"]:
    for resource in result["resources"]:
        download_url = resource.get("download_url") or resource.get("url")
        if "gpkg" in download_url:
            file_name = os.path.basename(download_url)
            local_path = os.path.join(SAVE_DIR, file_name)

            if os.path.exists(local_path):
                print(f"Skipping already downloaded: {file_name}")
                continue

            # Skip large files
            try:
                head = requests.head(download_url, allow_redirects=True, timeout=10)
                content_length = head.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_FILE_SIZE:
                    print(f"Skipping {file_name} — too large ({int(content_length) / (1024 ** 3):.2f} GB)")
                    continue
            except Exception as e:
                print(f"Could not retrieve size for {file_name}, skipping: {e}")
                continue

            # Download .gpkg file
            try:
                r = requests.get(download_url)
                with open(local_path, "wb") as f:
                    f.write(r.content)
                print(f"Downloaded: {local_path}")
            except Exception as e:
                print(f"Failed to download {download_url}: {e}")
                continue

            try:
                title = ""
                description = ""
                keywords = []
                for lang in LANGUAGES:
                    if lang in result["title"] and len(result["title"][lang]) > 0:
                        title = result["title"][lang]
                    if lang in result["description"] and len(result["description"][lang]) > 0:
                        description = result["description"][lang]
                    if lang in result["keywords"] and len(result["keywords"][lang]) > 0:
                        keywords = result["keywords"][lang]
                title = translate_text(title)
                description = translate_text(description)
                keywords = [translate_text(k) for k in keywords]

                # Save metadata
                with open(CSV_PATH, "a", newline='', encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["filename", "filetype", "short_description", "description", "keywords"])
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerow({
                        "filename": download_url,
                        "filetype": 'gpkg',
                        "short_description": title,
                        "description": description,
                        "keywords": "; ".join(keywords)
                    })

            except Exception as e:
                print(f"Failed to save {download_url}: {e}")
                continue

            gc.collect()
