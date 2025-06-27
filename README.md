# embedding-geospatial-modalities
Summer@EPFL project

# Installation
Tested with Ubuntu (WSL).
```
pip install -r requirements.txt
```

## Build s2geometry
Source: https://github.com/google/s2geometry?tab=readme-ov-file#build
Commented out include ```<absl/base/thread_annotations.h>``` in all files.
To build and test using bazel, from within third_party/s2geometry/src, run:
```
bazel test "//:*"
```
To build the libary without testing, from within third_party/s2geometry/src, run:
```
bazel build //:s2
```

## Use MAE
Source: https://github.com/facebookresearch/mae
Make the following fix to timm==0.3.2: https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842
Run from within third_party/mae:
```
export PYTHONPATH="$PWD:$PYTHONPATH"
```

# File Information
```scrape.py```: Uses opendata.swiss API to download all ```.gpkg``` files to gpkg_files and save their descriptions to ```geospatial_swiss.csv```.
```translate.py```: Unzips all the files in the gpkg_files folder and only keeps the ```.gpkg```. Translates all files into English and saves the translated file to the gpkg_files/translated folder.
```autogenerate_metadata.py```: Builds a CSV file with file names, layer names, and cell IDs as well as their corresponding features, spatial data, and natural language descriptions of the metadata. The CSV is used for training.
```s2patches.py```: Uses s2geometry to encode ```.gpkg``` files into patches. Results are saved to the patches folder.
```train.py```: Trains a MaskedAutoencoderViT to encode patches. TODO: Train an LM to encode text and build a cross-modal embedding model that will link text queries / metadata to patches.
```qdrant_retrieval.py```: Retrieves the closest matching patches given a text query.
```demo.py```: Using the retrieved patches and cross-modal model, extract metadata from the patches to answer user prompts.
