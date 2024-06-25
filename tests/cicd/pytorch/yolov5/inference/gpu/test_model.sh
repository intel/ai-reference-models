#!/bin/bash
set -e

echo "Setup IPEX-XPU Test Enviroment for Yolov5 Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/yolov5/inference/gpu/output/${PRECISION}"}
is_lkg_drop=$2
platform=$3
DATASET_DIR=$4

if [[ "${platform}" == "flex=gpu" || "${platform}" == "ATS-M" ]]; then
    runner="Flex"
    multi_tile="False"
elif [[ "${platform}" == "max-gpu" || "${platform}" == "pvc" ]]; then
    runner="Max"
    multi_tile=${MULTI_TILE}
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
cd models_v2/pytorch/yolov5/inference/gpu

python3 -m pip install -r requirements.txt
export PYTHONPATH=$(pwd)/../../../../common

OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} DATASET_DIR=${DATASET_DIR} PLATFORM=${runner} MODEL_NAME=yolov5m ./run_model.sh
cd -
