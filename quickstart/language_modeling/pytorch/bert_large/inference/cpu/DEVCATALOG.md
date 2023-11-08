# PyTorch BERT Large inference

## Description 
This document has instructions for running BERT Large inference using Intel Extension for PyTorch. 

## Pull Command

Docker image based on CentOS Stream8
```
docker pull intel/language-modeling:pytorch-cpu-centos-bert-large-inference
```

Docker image based on Ubuntu 22.04
```
docker pull intel/language-modeling:pytorch-cpu-ubuntu-bert-large-inference
```
## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `run_multi_instance_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, avx-fp32, int8, avx-int8, bf32 or bf16) using the [huggingface fine tuned model](https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin). |
| `run_multi_instance_throughput.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, avx-fp32, int8, avx-int8,bf32 or bf16) using the [huggingface fine tuned model](https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin). |
| `run_accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16) using the [huggingface fine tuned model](https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin). |

**Note:** The `avx-int8` and `avx-fp32` precisions run the same scripts as `int8` and `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets
Please following this [link](https://github.com/huggingface/transformers/tree/v3.0.2/examples/question-answering) to get `dev-v1.1.json` and set the `EVAL_DATA_FILE` environment variable to point to the file:
```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
export EVAL_DATA_FILE=$(pwd)/dev-v1.1.json
```
## Pre-Trained Model
Download the `config.json` and fine tuned model from huggingface and set the `PRETRAINED_MODEL` environment variable to point to the directory that has both files:
```
mkdir bert_squad_model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
PRETRAINED_MODEL=$(pwd)/bert_squad_model
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
To run the BERT Large inference scripts, set environment variables to specify the dataset directory, precision and an output directory. 

```
export OS=<provide either centos or ubuntu>
export EVAL_DATA_FILE=<path to the eval data>
export OUTPUT_DIR=<directory where log files will be written>
export PRECISION=<specify the precision>
export SCRIPT=quickstart/<specify the quickstart script>
export PRETRAINED_MODEL=<path to pre-trained model>

IMAGE_NAME=intel/language-modeling:pytorch-cpu-${OS}-bert-large-inference
DOCKER_ARGS="--privileged --init -it"
WORKDIR=/workspace/pytorch-bert-large-inference
EVAL_SCRIPT="${WORKDIR}/quickstart/transformers/examples/legacy/question-answering/run_squad.py"

docker run --rm \
  --env PRECISION=${PRECISION} \
  --env EVAL_DATA_FILE=${EVAL_DATA_FILE} \
  --env EVAL_SCRIPT=${EVAL_SCRIPT} \
  --env FINETUNED_MODEL=${PRETRAINED_MODEL} \
  --env INT8_CONFIG=${WORKDIR}/quickstart/configure.json \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${EVAL_DATA_FILE}:${EVAL_DATA_FILE} \
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

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/quickstart/language_modeling/pytorch/bert_large/inference/cpu/DEVCATALOG.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pyt-cpu)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
