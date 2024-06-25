#!/bin/bash
set -e

echo "Setup IPEX-XPU Test Enviroment for DLRM v1 Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/dlrm/inference/gpu/output/${PRECISION}"}
is_lkg_drop=$2
platform=$3
DATASET_DIR=$4
MULTI_TILE=$5

if [[ "${platform}" == "flex=gpu" || "${platform}" == "ATS-M" ]]; then
    runner="Flex"
    multi_tile="False"
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
cd models_v2/pytorch/pytorch/dlrm/inference/gpu
./setup.sh

if [[ ! -d "checkpoint_dir" ]]; then
    mkdir -p checkpoint_dir
    cd checkpoint_dir
    ./bench/dlrm_s_criteo_kaggle.sh [--test-freq=1024]
    cd -
fi

OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} DATASET_DIR=${DATASET_DIR} CKPT_DIR="checkpoint_dir" MULTI_TILE=${multi_tile} PLATFORM=Flex ./run_model.sh
cd -
