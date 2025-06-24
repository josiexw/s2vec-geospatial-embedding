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
Run from within third_party/mae
```
export PYTHONPATH="$PWD:$PYTHONPATH"
```
