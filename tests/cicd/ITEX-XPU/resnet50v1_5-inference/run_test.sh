#!/bin/bash
set -e

echo "Setup ITEX-XPU Test Enviroment for ResNet50v1.5 Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/ITEX-XPU/resnet50v1_5-inference/${PRECISION}"}
is_lkg_drop=$2
DATASET=$3

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
cd models_v2/tensorflow/resnet50v1_5/inference/gpu
./setup.sh

# Download PB file:
if [[ "${PRECISION}" == "int8" ]]; then
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_1/resnet50_v1_int8.pb
  PB_FILE=$(pwd)/resnet50_v1_int8.pb
elif [[ "${PRECISION}" == "float32" || "${PRECISION}" == "tensorflow32" || "${PRECISION}" == "float16" || "${PRECISION}" == "bfloat16" ]]; then
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_1/resnet50_v1.pb
  PB_FILE=$(pwd)/resnet50_v1.pb
fi

OUTPUT_DIR=${OUTPUT_DIR} DTYPE=${PRECISION} PB_FILE_PATH=${PB_FILE} DATASET_DIR=${DATASET} TEST_MODE=inference ./run_model.sh
cd - 
