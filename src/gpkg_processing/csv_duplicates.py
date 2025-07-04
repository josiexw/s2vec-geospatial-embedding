import pandas as pd

df = pd.read_csv("opendataswiss.parquet.csv")
df = df.drop_duplicates(subset="filename")
df["filename"] = df["filename"].apply(lambda x: x.split("/")[-1].replace(".zip", ""))
df.to_csv("opendataswiss.parquet.csv", index=False)
