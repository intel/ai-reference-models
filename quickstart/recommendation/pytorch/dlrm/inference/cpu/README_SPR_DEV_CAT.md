# PyTorch DLRM inference

## Description 
This document has instructions for running DLRM Inference using Intel Extension for PyTorch. 

## Pull Command
```
docker pull intel/recommendation:pytorch-spr-dlrm-inference
```

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `inference_performance.sh` | Run inference to verify performance for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16). |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16). |

**Note:**The `avx-int8` and `avx-fp32` precisions run the same scripts as `int8` and `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets
### Criteo Terabyte Dataset

The [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) is used to run DLRM. To download the dataset, you will need to visit the Criteo website and accept their terms of use: [https://labs.criteo.com/2013/12/download-terabyte-click-logs/](https://labs.criteo.com/2013/12/download-terabyte-click-logs/). Copy the download URL into the command below as the `<download url>` and replace the `<dir/to/save/dlrm_data>` to any path where you want to download and save the dataset.
```
export DATASET_DIR=<dir/to/save/dlrm_data>
mkdir ${DATASET_DIR} && cd ${DATASET_DIR}
curl -O <download url>/day_{$(seq -s , 0 23)}.gz
gunzip day_*.gz
```
The raw data will be automatically preprocessed and saved as `day_*.npz` to the `DATASET_DIR` when DLRM is run for the first time. On subsequent runs, the scripts will automatically use the preprocessed data.

## Pre-trained Model
Download the DLRM PyTorch weights (`tb00_40M.pt`, 90GB) from the [MLPerf repo](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch#more-information-about-the-model-weights) and set the `WEIGHT_PATH` to point to the weights file.
```
export WEIGHT_PATH=<path to the tb00_40M.pt file>
```

## Docker Run
(Optional) Export related proxy into docker environment.
```bash
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```
To run DLRM inference, set environment variables to specify the dataset directory, precision,pre-trained model, and an output directory. 
```
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export SCRIPT=quickstart/<specify the script to run>
export PRECISION=<specify the precision to run>
export BERT_WEIGHT=<specify path to pre-trained model>

IMAGE_NAME=intel/recommendation:pytorch-spr-dlrm-inference
DOCKER_ARGS="--privileged --init -it"
WORKDIR=/workspace/pytorch-spr-dlrm-inference

docker run --rm \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env WEIGHT_PATH=${WEIGHT_PATH} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${WEIGHT_PATH}:${WEIGHT_PATH} \
  --shm-size 8G \
  -w ${WORKDIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
  ```

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/recommendation)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/quickstart/recommendation/pytorch/dlrm/inference/cpu/README_SPR_DEV_CAT.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/dockerfiles/pytorch)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)