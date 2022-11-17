<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* git
* numactl
* wget
* IPython[all]
* Pillow>=8.1.2
* cython
* h5py
* imgaug
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/)
* keras==2.0.8
* matplotlib
* numpy==1.16.3
* opencv-python
* pycocotools
* scikit-image
* scipy==1.2.0

After installing the prerequisites, download & untar the model package.
Clone the [MaskRCNN repo](https://github.com/matterport/Mask_RCNN) and
download the [pretrained model](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)
to the same directory. Set environment variables for the path to your
`DATASET_DIR`, `MODEL_SRC_DIR` and an `OUTPUT_DIR` where log files will
be written, then run a [quickstart script](#quick-start-scripts).

```
wget <package url>
tar -xzf <package name>
cd <package dir>

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export MODEL_SRC_DIR=<path to the Mask RCNN models repo>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

git clone https://github.com/matterport/Mask_RCNN.git ${MODEL_SRC_DIR}
pushd ${MODEL_SRC_DIR}
wget -q https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
popd

./quickstart/<script name>.sh
```
