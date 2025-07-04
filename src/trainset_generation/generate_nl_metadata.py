import pandas as pd
import os

# === Config ===
FILE_CSV = './data/opendataswiss.parquet.csv'
METADATA_CSV = './data/patch_metadata.csv'
OUTPUT_CSV = './data/patch_metadata_pairs.csv'

METADATA_TEMPLATES = [
    """
    Has features {features};
    located at coordinates {coordinates} respectively;
    the features are described as {feature_info}.
    """,
    "{short_description}: {description}",
    """
    This patch is part of a dataset on {short_description}.
    It includes features {features} located at coordinates {coordinates}.
    """
]

def csv_exists():
    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=[
            "filename", "layer", "level", "cell_id", "metadata_text"
        ]).to_csv(OUTPUT_CSV, index=False)

def get_patch_metadata_pairs(file_df, metadata_df):
    metadata_pairs = []
    metadata_df = metadata_df.merge(file_df, on="filename", how="left")

    for _, row in metadata_df.iterrows():
        for template in METADATA_TEMPLATES:
            try:
                text = template.format(
                    filename=row["filename"],
                    layer_name=row["layer"],
                    level=row["level"],
                    cid=row["cell_id"],
                    features=row.get("features", ""),
                    coordinates=row.get("coordinates", ""),
                    addresses=row.get("addresses", ""),
                    feature_info=row.get("feature_info", ""),
                    short_description=row.get("short_description", ""),
                    description=row.get("description", "")
                )
                metadata_pairs.append({
                    "filename": row["filename"],
                    "layer": row["layer"],
                    "level": row["level"],
                    "cell_id": row["cell_id"],
                    "metadata_text": text.strip()
                })
            except Exception as e:
                print(f"Skipping template due to formatting error: {e}")

    return metadata_pairs

def main():
    csv_exists()
    file_df = pd.read_csv(FILE_CSV, usecols=["filename", "filetype", "short_description", "description", "keywords"])
    metadata_df = pd.read_csv(METADATA_CSV, usecols=["filename", "layer", "level",
                                                     "cell_id", "features", "coordinates",
                                                     "addresses", "feature_info"])
    rows = get_patch_metadata_pairs(file_df, metadata_df)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, mode='a', index=False, header=False)

if __name__ == "__main__":
    main()
