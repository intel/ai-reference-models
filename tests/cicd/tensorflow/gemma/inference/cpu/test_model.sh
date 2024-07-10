#!/bin/bash
set -e

echo "Setup TF-CPU Test Enviroment for GEMMA Keras Inference"

DATATYPE=$1
OUT_DIR=${OUT_DIR-"$(pwd)/tests/cicd/tensorflow/gemma/inference/cpu/output/${DATATYPE}"}
is_lkg_drop=$2
MODEL_DIR=$3
MAX_LEN=$4
BACKEND=$5

# Create the output directory in case it doesn't already exist
mkdir -p ${OUT_DIR}

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
cd models_v2/tensorflow/gemma/inference/cpu
./setup.sh

OUTPUT_DIR=${OUT_DIR} PRECISION=${DATATYPE} MODEL_PATH=${MODEL_DIR} MAX_LENGTH=${MAX_LEN} KERAS_BACKEND=${BACKEND} ./run_model.sh
cd -
