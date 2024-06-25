#!/bin/bash
set -e

echo "Setup ITEX-XPU Test Enviroment for Bert Large Training"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/tensorflow/bert_large/training/gpu/output/${PRECISION}"}
is_lkg_drop=$2
DATASET=$3
MULTI_TILE=$4
NUM_DEVICES=$5

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
cd models_v2/tensorflow/bert_large/training/gpu
./setup.sh
pip uninstall horovod
pip install intel-optimization-for-horovod

RESULTS_DIR=${OUTPUT_DIR} DATATYPE=${PRECISION} MULTI_TILE=${MULTI_TILE} DATA_DIR=${DATASET} NUM_DEVICES=${NUM_DEVICES} ./run_model.sh
cd -
