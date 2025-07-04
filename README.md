# embedding-geospatial-modalities
Summer@EPFL project <br />
MaskedAutoEncoderViT trained on patches with levels [10, 12, 14, 16, 18]. Use contrastive learning to 
correlate text query with patch and patch metadata.

# Installation
Tested with Ubuntu (WSL).
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Build s2geometry (OPTIONAL: code uses s2sphere)
Source: https://github.com/google/s2geometry?tab=readme-ov-file#build <br />
Commented out include ```<absl/base/thread_annotations.h>``` in all files. <br />
To build and test using bazel, from within third_party/s2geometry/src, run:
```
bazel test "//:*"
```
To build the libary without testing, from within third_party/s2geometry/src, run:
```
bazel build //:s2
```

## Use MAE
Source: https://github.com/facebookresearch/mae <br />
Make the following fix to timm==0.3.2: https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842 <br />
Run from within third_party/mae:
```
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Generating metadata ```autogenerate_metadata.py```
Source: https://github.com/mediagis/nominatim-docker/tree/master/4.2 <br />
Use Nominatim docker 4.2 to get the addresses from the coordinates without rate limits. <br />
To create the docker container (for Switzerland coordinates):
```
docker run -it
-e PBF_URL=https://download.geofabrik.de/europe/switzerland-latest.osm.pbf
-e REPLICATION_URL=Index of /europe/switzerland-updates
-p 8080:8080
--name nominatim
mediagis/nominatim:4.2
```
To start the docker container:
```
docker start -ai nominatim
```
To stop the docker container:
```
docker stop nominatim
```

# File Information
```scrape.py```: Uses opendata.swiss API to download all ```.gpkg``` files to ```/gpkg_files``` and save their filename, title (short_description), description, and keywords to ```opendataswiss.parquet.csv```. <br />
```translate.py``` (OPTIONAL): Unzips all the files in the ```/gpkg_files``` folder and only keeps the ```.gpkg```. Translates all files into English and saves the translated file to the gpkg_files/translated folder. <br />
```autogenerate_metadata.py```: Builds a CSV file ```patch_metadata.csv``` with file names, layer names, and cell IDs as well as their corresponding features, coordinates, addresses, and feature information. The CSV is used for training. <br />
```s2patches.py```: Uses s2geometry (or the Python implementation s2sphere) to encode ```.gpkg``` files into patches. Results are saved to the patches folder, and the patch information is saved to ```patch_embeddings.csv```. <br />
```train.py```: Trains a MaskedAutoencoderViT to encode patches. TODO: Train an LM to encode text and build a cross-modal embedding model that will link text queries / metadata to patches. <br />
```qdrant_retrieval.py```: Retrieves the closest matching patches given a text query. <br />
```demo.py```: Using the retrieved patches and cross-modal model, extract metadata from the patches to answer user prompts.