# Running ResNet50_v1.5 Inference on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch*

## Description 

This document has instructions for running ResNet50 V1.5 Inference using Intel® Extension for PyTorch on Intel®Max Series GPU. 

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)  |
| Drivers | GPU-compatible drivers need to be installed: Download [Driver](https://dgpu-docs.intel.com/driver/installation.html) |
| Software | Docker* |

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Runs ResNet50 V1.5 INT8,FP16,BF16,FP32 and TF32 inference on single tile and two tiles|

## Datasets

The [ImageNet](http://www.image-net.org/) validation dataset is used.

Download and extract the ImageNet2012 dataset from http://www.image-net.org/, then move validation images to labeled subfolders, using [the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

After running the data prep script, your folder structure should look something like this:

```
imagenet
└── val
    ├── ILSVRC2012_img_val.tar
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   ├── ILSVRC2012_val_00006697.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` directory should be set as the
`DATASET_DIR`
(for example: `export DATASET_DIR=/home/<user>/imagenet`).

> [!NOTE]
> The `run_model.sh` can also run inference on dummy data. In this case, `DATASET_DIR` does not have to be set or volume mounted to the container. 

### Docker pull command:

```
docker pull intel/image-recognition:pytorch-max-gpu-resnet50v1-5-inference
```
The ResNet50 v1.5 inference container includes scripts, model and libraries needed to run INT8,FP16,BF16,FP32 and TF32 inference. To run the `run_model.sh` quickstart script using this container, you'll need to set the environment variable and provide volume mounts for the ImageNet dataset if real dataset is required. Otherwise, the script uses dummy data. You will need to provide an output directory where log files will be written. 

```bash
#Optional
export PRECISION=<provide either INT8,BF16,FP16,FP32 or TF32, otherwise (default: INT8)>
export BATCH_SIZE=<provide batch size, otherwise (default: 1024)>
export NUM_ITERATIONS=<provide number of iterations,otherwise (default: 500)>
export DATASET_DIR=<path to ImageNet dataset>

#Required
export OUTPUT_DIR=<path to output logs directory>
export PLATFORM=Max
export MULTI_TILE=<provide True of False to enable/disable multi-tile inference>

IMAGE_NAME=intel/image-recognition:pytorch-max-gpu-resnet50v1-5-inference
DOCKER_ARGS="--rm -it"
SCRIPT=run_model.sh

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env PLATFORM=${PLATFORM} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MULTI_TILE=${MULTI_TILE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
```
## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/max-gpu)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
