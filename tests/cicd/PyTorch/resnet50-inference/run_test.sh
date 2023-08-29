#!/bin/bash
set -e

echo "Setup PyTorch Test Enviroment for ResNet50 Inference"

PRECISION=$1
SCRIPT=$2
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/PyTorch/resnet50-inference/${SCRIPT}/${PRECISION}"}
WORKSPACE=$3
is_lkg_drop=$4
DATASET=$5

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  export PATH=${WORKSPACE}/miniconda3/bin:$PATH
  source ${WORKSPACE}/pytorch_setup/setvars.sh
  source ${WORKSPACE}/pytorch_setup/compiler/latest/env/vars.sh
  source ${WORKSPACE}/pytorch_setup/mkl/latest/env/vars.sh
  source ${WORKSPACE}/pytorch_setup/tbb/latest/env/vars.sh
  source ${WORKSPACE}/pytorch_setup/mpi/latest/env/vars.sh
  conda activate pytorch
fi

# Run script
OUTPUT_DIR=${OUTPUT_DIR} DATASET_DIR=${DATASET} ./quickstart/image_recognition/pytorch/resnet50/inference/cpu/${PRECISION}/${SCRIPT}
