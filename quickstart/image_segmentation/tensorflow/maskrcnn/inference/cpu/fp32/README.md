<!--- 0. Title -->
# Mask R-CNN FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running Mask R-CNN FP32 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[maskrcnn-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/maskrcnn-fp32-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets and Pretrained Model

Download the [MS COCO 2014 dataset](http://cocodataset.org/#download).
Set the `DATASET_DIR` to point to this directory when running Mask R-CNN.
```
# Create a new directory, to be set as DATASET_DIR
mkdir $DATASET_DIR
cd $DATASET_DIR

# Download and extract MS COCO 2014 dataset
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
cp annotations/instances_val2014.json annotations/instances_minival2014.json

export DATASET_DIR=${DATASET_DIR}
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [fp32_inference.sh](fp32_inference.sh) | Runs inference with batch size 1 using Coco dataset and pretrained model|

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
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/maskrcnn-fp32-inference.tar.gz
tar -xzf maskrcnn-fp32-inference.tar.gz
cd maskrcnn-fp32-inference

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export MODEL_SRC_DIR=<path to the Mask RCNN models repo>

git clone https://github.com/matterport/Mask_RCNN.git ${MODEL_SRC_DIR}
pushd ${MODEL_SRC_DIR}
wget -q https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
popd

./quickstart/<script name>.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts,
[pretrained model](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5),
and libraries needed to run  Mask R-CNN FP32 inference. To run one
of the quickstart scripts  using this container, you'll need to provide
volume mounts for the dataset and an output directory.

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/image-segmentation:tf-1.15.2-maskrcnn-fp32-inference \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

