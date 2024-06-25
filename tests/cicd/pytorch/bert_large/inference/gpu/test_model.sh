#!/bin/bash
set -e

echo "Setup IPEX-XPU Test Enviroment for Bert Large Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/bert_large/inference/gpu/output/${PRECISION}"}
is_lkg_drop=$2
platform=$3
DATASET_DIR=$4
MULTI_TILE=$5

if [[ "${platform}" == "flex=gpu" || "${platform}" == "ATS-M" ]]; then
    exit 1
elif [[ "${platform}" == "max-gpu" || "${platform}" == "pvc" ]]; then
    runner="Max"
    multi_tile=${MULTI_TILE}
elif [[ "${platform}" == "arc" ]]; then
    runner="Arc"
    multi_tile="False"
    if [[ "${PRECISION}" != "FP16" ]]; then
        exit 1
    fi
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
else
  source /oneapi/compiler/latest/env/vars.sh
  source /oneapi/mpi/latest/env/vars.sh
  source /oneapi/mkl/latest/env/vars.sh
  source /oneapi/tbb/latest/env/vars.sh
  source /oneapi/ccl/latest/env/vars.sh
fi

# run following script
cd models_v2/pytorch/bert_large/inference/gpu
# Download pretrain model
if [[ ! -d "squad_large_finetuned_checkpoint" ]]; then
    mkdir -p squad_large_finetuned_checkpoint
    wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json -O squad_large_finetuned_checkpoint/config.json
    wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/pytorch_model.bin -O squad_large_finetuned_checkpoint/pytorch_model.bin
    wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json -O squad_large_finetuned_checkpoint/tokenizer.json
    wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer_config.json -O squad_large_finetuned_checkpoint/tokenizer_config.json
    wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt -O squad_large_finetuned_checkpoint/vocab.txt
fi

BERT_WEIGHT=squad_large_finetuned_checkpoint
./setup.sh

OUTPUT_DIR=${OUTPUT_DIR} BERT_WEIGHT="squad_large_finetuned_checkpoint" PRECISION=${PRECISION} DATASET_DIR=${DATASET_DIR} MULTI_TILE=${multi_tile} PLATFORM=${runner} ./run_model.sh
cd -
