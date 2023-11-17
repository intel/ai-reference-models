# Running PyTorch GPT-J 6B Inference on Intel® Max Series GPU.

## Description 

This document has instructions for running GPT-J 6B Inference with FP16 and INT4 precision using Intel® Extension for PyTorch on Intel® Max Series GPU. 

## Datasets

Use the [download_cnndm.py](../../../../../../models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu/download_cnndm.py) to download and pre-process the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset. Set `DATASET_DIR` to point to directory of the dataset. Please install the following components to download the dataset. 

```bash
pip install virtualenv
python<version> -m venv <virtual-environment-name>
source env/bin/activate
pip install accelerate datasets evaluate nltk rouge_score simplejson transformers
```
## Configuration Files

Download the required configuration files [calibration-list.txt](https://github.com/mlcommons/inference/blob/v3.1/language/gpt-j/calibration-list.txt) and [mlperf.conf](https://github.com/mlcommons/inference/blob/v3.1/mlperf.conf). The `calibration-list.txt` file is used for calibration and required only if you are using `INT4` precision for inference. Download the files to `CONFIG_DIR` directory. 

## Pre-trained Model
 Use the following instructions to download the model.
```bash
export PRETRAINED_MODEL_DIR=<path to the pre-trained model directory>
wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download --output-document checkpoint.zip
unzip checkpoint.zip
mv gpt-j/checkpoint-final/* ${PRETRAINED_MODEL_DIR}
rm -rf gpt-j
```
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Runs GPT-J 6B inference with float16 and int4 precisions |

Requirements:
* Host machine has Intel(R) Data Center Max Series 1550 GPU x4 OAM GPU
* Follow instructions to install GPU-compatible [driver](https://dgpu-docs.intel.com/driver/installation.html)
* Docker

### Docker pull command:

```
docker pull intel/generative-ai:pytorch-max-gpu-gptj-6b-inference
```
The GPT-J 6B inference container includes scripts,model and libraries need to run FP16 and INT4 inference. To run the `inference.sh` quickstart script using this container, you'll need to set the environment variable and provide volume mounts for the dataset. You will need to provide an output directory where log files will be written. 

```bash
export DATASET_DIR=<path to the dataset directory>
export PRETRAINED_MODEL_DIR=<path to the pre-trained model directory>
export OUTPUT_DIR=<path to output directory>
export CONFIG_DIR=<path to config directory>
export NUM_GPU_TILES=<provide 8 for x4 OAM Module>
export PRECISION=<provide either float16 or int4 precision>
export INFERENCE_MODE=<provide either Server or Offline inference mode>
export INFERENCE_TYPE=<provide either accuracy or benchmark>
export ACCURACY=<provide true for accuracy evaluation and false for performance.Default is true>
export BATCH_SIZE=<export batch size. For Server mode, default is 32 and for offline mode default is 1>

IMAGE_NAME=intel/generative-ai:pytorch-max-gpu-gptj-6b-inference
SCRIPT=quickstart/inference.sh
DOCKER_ARGS="--rm -it"

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env NUM_GPU_TILES=${NUM_GPU_TILES} \
  --env ACCURACY=${ACCURACY} \
  --env INFERENCE_MODE=${INFERENCE_MODE} \
  --env INFERENCE_TYPE=${INFERENCE_TYPE} \
  --env PRETRAINED_MODEL_DIR=${PRETRAINED_MODEL_DIR} \
  --env CONFIG_DIR=${CONFIG_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${PRETRAINED_MODEL_DIR}:${PRETRAINED_MODEL_DIR} \
  --volume ${CONFIG_DIR}:${CONFIG_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
```
