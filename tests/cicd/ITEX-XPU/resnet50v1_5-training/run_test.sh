#!/bin/bash
set -e

echo "Setup ITEX-XPU Test Enviroment for ResNet50v1.5 Training"

CONFIG_FILE=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/ITEX-XPU/resnet50v1_5-training/${PRECISION}"}
is_lkg_drop=$2
DATASET=$3
MULTI_TILE=$4

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/tensorflow_setup/bin/activate tensorflow
else
  source /oneapi/compiler/latest/env/vars.sh
  source /oneapi/mpi/latest/env/vars.sh
  source /oneapi/mkl/latest/env/vars.sh
  source /oneapi/tbb/latest/env/vars.sh
  source /oneapi/ccl/latest/env/vars.sh
fi

# run following script
cd models_v2/tensorflow/resnet50v1_5/training/gpu
./setup.sh

apt-get update 

pip install intel-optimization-for-horovod

OUTPUT_DIR=${OUTPUT_DIR} CONFIG_FILE=$(pwd)/${CONFIG_FILE} DATASET_DIR=${DATASET} MULTI_TILE=${MULTI_TILE} ./run_model.sh
cd - 
