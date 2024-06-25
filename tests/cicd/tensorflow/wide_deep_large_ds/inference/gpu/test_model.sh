#!/bin/bash
set -e

echo "Setup ITEX-XPU Test Enviroment for Wide Deep Large Inference"

OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/tensorflow/wide_deep_large_ds/inference/gpu/${PRECISION}"}
is_lkg_drop=$2
DATASET=$1

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
cd models_v2/tensorflow/wide_deep_large_ds/inference/gpu
./setup.sh

# Download the PB File:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_0/wide_deep_fp16_pretrained_model.pb
PB_FILE_PATH=$(pwd)/wide_deep_fp16_pretrained_model.pb

OUTPUT_DIR=${OUTPUT_DIR} PB_FILE_PATH=${PB_FILE_PATH} DATASET_PATH=${DATASET} ./run_model.sh
cd -
