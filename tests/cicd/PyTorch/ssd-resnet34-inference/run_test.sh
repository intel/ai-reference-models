#!/bin/bash
set -e

echo "Setup PyTorch Test Enviroment for SSD-ResNet34 Inference"

PRECISION=$1
SCRIPT=$2
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/PyTorch/ssd-resnet34-inference/${SCRIPT}/${PRECISION}"}
WORKSPACE=$3
is_lkg_drop=$4
DATASET=$5

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  #export PATH=${WORKSPACE}/miniconda3/bin:$PATH
  #source ${WORKSPACE}/pytorch_setup/setvars.sh
  #source ${WORKSPACE}/pytorch_setup/compiler/latest/env/vars.sh
  #source ${WORKSPACE}/pytorch_setup/mkl/latest/env/vars.sh
  #source ${WORKSPACE}/pytorch_setup/tbb/latest/env/vars.sh
  #source ${WORKSPACE}/pytorch_setup/mpi/latest/env/vars.sh
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
  #conda activate pytorch
fi

# run model specific dependencies:
pip install matplotlib Pillow defusedxml
pip install pycocotools

# Download Pretrained Model:
export CHECKPOINT_DIR=$(pwd)/tests/cicd/PyTorch/ssd-resnet34-inference/
./quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/download_model.sh

# Download dataset
export DATASET_DIR=$(pwd)/tests/cicd/PyTorch/ssd-resnet34-inference/
./quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/download_dataset.sh

# Run script
OUTPUT_DIR=${OUTPUT_DIR} CHECKPOINT_DIR=${CHECKPOINT_DIR} DATASET_DIR=${DATASET_DIR} ./quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/${SCRIPT} ${PRECISION}
