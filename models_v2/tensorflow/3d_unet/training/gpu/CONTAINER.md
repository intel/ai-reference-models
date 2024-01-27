# TensorFlow UNet-3D Training

## Description 
This document has instructions for running UNet-3D training with BFloat16 precision using Intel® Extension for TensorFlow on Intel® Data Center GPU Max Series.

## Datasets

Download the dataset by registering on [Brain Tumor Segmentation 2019 dataset](https://www.med.upenn.edu/cbica/brats-2019/). After downloading dataset, follow steps in the [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_3D_Medical#quick-start-guide) to pre-process the dataset in tfecords format. 

Set the `DATASET_DIR` to point to the TF records directory when running UNet-3D training. 

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Runs BFloat16 Training on single and two tiles |

Requirements:

* Host has [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)
* Host has installed latest Intel® Data Center GPU Max Series [Drivers](https://dgpu-docs.intel.com/driver/installation.html)
* Docker

### Docker pull command:

```bash
docker pull intel/image-segmentation:tf-max-gpu-unet-3d-training
```
The UNet-3D training container includes scripts, model and libraries needed to run BFloat16 Training. To run the quickstart scripts using this container, you'll need to provide volume mounts of the Brain Tumor Segmentation dataset. You will also need to provide an output directory to store logs.

```bash
#Required
export DATASET_DIR=<provide path to pre-processed dataset>
export PRECISION=bfloat16
export MULTI_TILE=<provide True for multi-tile training, False for single-tile training>
export OUTPUT_DIR=<provide path to output logs directory>
SCRIPT=run_model.sh

#Optional
export BATCH_SIZE=<provide batch size. Otherwise default is 1>
IMAGE_NAME=intel/image-segmentation:tf-max-gpu-unet-3d-training

docker run -it \
--device /dev/dri \
--env BATCH_SIZE=${BATCH_SIZE} \
--env MULTI_TILE=${MULTI_TILE} \
--env DATASET_DIR=${DATASET_DIR} \
--env PRECISION=${PRECISION} \
--env OUTPUT_DIR=${OUTPUT_DIR} \
--env http_proxy=${http_proxy} \
--env https_proxy=${https_proxy} \
--env no_proxy=${no_proxy} \
--volume ${DATASET_DIR}:${DATASET_DIR} \
--volume /dev/dri:/dev/dri \
--volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
${IMAGE_NAME} \
/bin/bash $SCRIPT
```
## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/max-gpu)

## Support
Support for Intel® Extension for TensorFlow* is available at [Intel® AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for TensorFlow* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
