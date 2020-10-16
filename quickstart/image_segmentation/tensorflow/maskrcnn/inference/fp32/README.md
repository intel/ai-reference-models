<!--- 0. Title -->
# Mask R-CNN FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running Mask R-CNN FP32 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[maskrcnn-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_0_0/maskrcnn-fp32-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets and Pretrained Model

1. Download the [MS COCO 2014 dataset](http://cocodataset.org/#download). 

    Set the `DATASET_DIR` to point to this directory when running Mask R-CNN.

2. Clone the [Mask R-CNN model repository](https://github.com/matterport/Mask_RCNN).
It is used as external model directory for dependencies. Download pre-trained COCO weights `mask_rcnn_coco.h5)` from the
[Mask R-CNN repository release page](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5),
and place it in the `MaskRCNN` directory.
```
$ https://github.com/matterport/Mask_RCNN.git
$ cd Mask_RCNN
$ wget -q https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
```

    Set `MODEL_SRC_DIR` to path to `MaskRCNN` directory


<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [fp32_inference.sh](fp32_inference.sh) | Runs inference with batch size 1 using Coco dataset and pretrained model|

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

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

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_0_0/maskrcnn-fp32-inference.tar.gz
tar -xzf maskrcnn-fp32-inference.tar.gz
cd maskrcnn-fp32-inference

quickstart/<script name>.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
Mask R-CNN FP32 inference. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory.

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>
MODEL_SRC_DIR=<path to the Mask RCNN models repo>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MODEL_SRC_DIR=${MODEL_SRC_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${MODEL_SRC_DIR}:${MODEL_SRC_DIR} \
  --privileged --init -t \
  amr-registry.caas.intel.com/aipg-tf/model-zoo:1.15.2-image-segmentation-maskrcnn-fp32-inference \
  /bin/bash quickstart/<script name>.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

