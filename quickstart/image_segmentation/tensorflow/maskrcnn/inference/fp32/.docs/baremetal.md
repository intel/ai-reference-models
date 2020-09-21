<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/)
* numactl
* pycocotools
* numpy==1.16.0
* scipy==1.2.0
* Pillow
* cython
* matplotlib
* scikit-image
* keras==2.0.8
* opencv-python
* h5py
* imgaug
* IPython[all]

After installing the prerequisites, download & untar the model package.
Set environment variables for the path to your `DATASET_DIR`, `MODEL_SRC_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).


```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>
MODEL_SRC_DIR=<path to the Mask RCNN models repo>

wget <package url>
tar -xzf <package name>
cd <package dir>

quickstart/<script name>.sh
```
