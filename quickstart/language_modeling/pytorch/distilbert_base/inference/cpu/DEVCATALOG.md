# PyTorch DistilBERT Base inference

## Description 
This document has instructions for running DistilBERT Base inference using Intel Extension for PyTorch. 

## Pull Command

Docker image based on CentOS Stream8
```
docker pull intel/language-modeling:pytorch-cpu-centos-distilbert-inference
```

Docker image based on Ubuntu 22.04
```
docker pull intel/language-modeling:pytorch-cpu-ubuntu-distilbert-inference
```

# Quick Start Scripts
|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | run_multi_instance_throughput.sh fp32 | run_multi_instance_realtime.sh fp32 | run_accuracy.sh fp32 |
| BF32        | run_multi_instance_throughput.sh bf32 | run_multi_instance_realtime.sh bf32 | run_accuracy.sh bf32 |
| BF16        | run_multi_instance_throughput.sh bf16 | run_multi_instance_realtime.sh bf16 | run_accuracy.sh bf16 |
| INT8-FP32        | run_multi_instance_throughput.sh int8-fp32 | run_multi_instance_realtime.sh int8-fp32 | run_accuracy.sh int8-fp32 |
| INT8-BF16       | run_multi_instance_throughput.sh int8-bf16 | run_multi_instance_realtime.sh int8-bf16 | run_accuracy.sh int8-bf16 |

# Datasets
Use the following instructions to download the SST-2 dataset.
Also, clone the AI Reference Models GitHub Repository and set the `MODEL_DIR` directory.
```
git clone https://github.com/IntelAI/models.git
cd models
git checkout v2.9.0
export MODEL_DIR=$(pwd)
cd -
export DATASET_DIR=$(pwd)
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip -O $DATASET_DIR/SST-2.zip
unzip $DATASET_DIR/SST-2.zip -d $DATASET_DIR/
python $MODEL_DIR/quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/convert.py $DATASET_DIR
```

# Pre-Trained Model
Follow the instructions below to download the pre-trained model. 

```
git clone https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
export PRETRAINED_MODEL=$(pwd)/distilbert-base-uncased-finetuned-sst-2-english
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
To run DistilBERT inference, set environment variables to specify the dataset directory, precision to run, model directory and an output directory. 
```
export OS=<provide either centos or ubuntu>
export HF_DATASETS_OFFLINE=<provide 0 or 1. providing 1 downloads the datasets"
export SEQUENCE_LENGTH=128
export CORE_PER_INSTANCE=<recommended core per instance are 4 for realtime inference and 32 for throughput inference and accuracy>
export PRECISION=<provide the precision>
export PRETRAINED_MODEL=<path to pre-trained model>
export DATASET_DIR=<path to dataset directory>
export OUTPUT_DIR=<path to output directory>

IMAGE_NAME=intel/language-modeling:pytorch-cpu-${OS}-distilbert-inference
SCRIPT=quickstart/<name of script>
DOCKER_ARGS="--privileged --init -it"
WORKDIR=/workspace/pytorch-distilbert-inference
EVAL_SCRIPT="${WORKDIR}/quickstart/transformers/examples/pytorch/text-classification/run_glue.py"

docker run --rm \
  --env PRECISION=${PRECISION} \
  --env EVAL_SCRIPT=${EVAL_SCRIPT} \
  --env HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE} \
  --env FINETUNED_MODEL=${PRETRAINED_MODEL} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env SEQUENCE_LENGTH=${SEQUENCE_LENGTH} \
  --env CORE_PER_INSTANCE=${CORE_PER_INSTANCE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${PRETRAINED_MODEL}:${PRETRAINED_MODEL} \
  --shm-size 8G \
  -w ${WORKDIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT ${PRECISION}
  ```
## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/language-modeling)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/DEVCATALOG.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pyt-cpu)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
