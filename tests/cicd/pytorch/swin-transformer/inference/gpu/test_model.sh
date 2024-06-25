#!/bin/bash
set -e

echo "Setup IPEX-XPU Test Enviroment for Swin transformer Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/swin-transformer/inference/gpu/output/${PRECISION}"}
is_lkg_drop=$2
platform=$3
DATASET_DIR=$4
MULTI_TILE=$5

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
cd models_v2/pytorch/swin-transformer/inference/gpu

source ./setup.sh

OUTPUT_DIR=${OUTPUT_DIR} DATASET_DIR=${DATASET_DIR} MULTI_TILE=${multi_tile} PLATFORM=${runner} ./run_model.sh
cd -
