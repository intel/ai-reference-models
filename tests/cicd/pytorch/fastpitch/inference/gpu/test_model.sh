#!/bin/bash
set -e

echo "Setup IPEX-XPU Test Enviroment for FastPitch Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/fastpitch/inference/gpu/output/${PRECISION}"}
is_lkg_drop=$2
platform=$3
MULTI_TILE=$4

if [[ "${platform}" == "flex=gpu" || "${platform}" == "ATS-M" ]]; then
    runner="Flex"
    multi_tile="False"
elif [[ "${platform}" == "max-gpu" || "${platform}" == "pvc" ]]; then
    exit 1
elif [[ "${platform}" == "arc" ]]; then
    exit 1
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
cd models_v2/pytorch/fastpitch/inference/gpu

# Download dataset
if [[ ! -d "LJSpeech-1.1" ]]; then
    wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    tar -xvf LJSpeech-1.1.tar.bz2
fi

DATASET_DIR=LJSpeech-1.1

# Download pretrain model
if [[ ! -d "pretrained_models" ]]; then
    bash scripts/download_models.sh fastpitch
    bash scripts/download_models.sh hifigan
    # Prepare mel files from groundtruth
    bash scripts/prepare_dataset.sh
fi

CKPT_DIR=pretrained_models
./setup.sh

OUTPUT_DIR=${OUTPUT_DIR} CKPT_DIR="pretrained_models" DATASET_DIR=${DATASET_DIR} MULTI_TILE=${multi_tile} PLATFORM=${runner} ./run_model.sh
cd -
