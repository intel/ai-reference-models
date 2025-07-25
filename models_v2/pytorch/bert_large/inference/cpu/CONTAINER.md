# PyTorch BERT Large inference

## Description 
This document has instructions for running BERT Large inference using Intel Extension for PyTorch. 

## Pull Command

```bash
docker pull intel/language-modeling:pytorch-cpu-bert-large-inference
```

**Note:** The `avx-int8` and `avx-fp32` precisions run the same scripts as `int8` and `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets
Please [follow](https://github.com/huggingface/transformers/tree/v3.0.2/examples/question-answering) the link to get `dev-v1.1.json` and set the `EVAL_DATA_FILE` environment variable to point to the file:

```bash
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
export EVAL_DATA_FILE=$(pwd)/dev-v1.1.json
```

## Pre-Trained Model
Download the `config.json` and fine tuned model from huggingface and set the `FINETUNED_MODEL` environment variable to point to the directory that has both files:

```bash
mkdir bert_squad_model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
FINETUNED_MODEL=$(pwd)/bert_squad_model
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

```bash
export EVAL_DATA_FILE=<path to the eval data>
export OUTPUT_DIR=<directory where log files will be written>
export PRECISION=<specify the precision>
export FINETUNED_MODELL=<path to pre-trained model>
export TEST_MODE=<provide either REALTIME,THROUGHPUT OR ACCURACY mode>
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16 (for FP16 precision)
export TORCH_INDUCTOR=0

IMAGE_NAME=intel/language-modeling:pytorch-cpu-bert-large-inference

docker run \
  --cap-add 'SYS_NICE' \
  -it \
  --env PRECISION=${PRECISION} \
  --env EVAL_DATA_FILE=${EVAL_DATA_FILE} \
  --env FINETUNED_MODEL=${FINETUNED_MODEL} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env TEST_MODE=${TEST_MODE} \
  --env TORCH_INDUCTOR=${TORCH_INDUCTOR} \
  --env DNNL_MAX_CPU_ISA=${DNNL_MAX_CPU_ISA} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${EVAL_DATA_FILE}:${EVAL_DATA_FILE} \
  --volume ${FINETUNED_MODEL}:${FINETUNED_MODEL} \
  ${DOCKER_RUN_ENVS} \
  $IMAGE_NAME \
  /bin/bash "run_model.sh"
  ```

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/language-modeling)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/models_v2/pytorch/bert_large/inference/cpu/CONTAINER.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pytorch)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
