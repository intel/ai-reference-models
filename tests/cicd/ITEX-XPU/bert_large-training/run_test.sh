#!/bin/bash
set -e

echo "Setup ITEX-XPU Test Enviroment for Bert Large Training"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/ITEX-XPU/bert_large-training/${PRECISION}"}
is_lkg_drop=$2
DATASET=$3
MULTI_TILE=$4

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/tensorflow_setup/bin/activate tensorflow
fi

# run following script
cd models_v2/tensorflow/bert_large/training/gpu
./setup.sh
pip uninstall horovod 
pip install intel-optimization-for-horovod

RESULTS_DIR=${OUTPUT_DIR} DATATYPE=${PRECISION} MULTI_TILE=${MULTI_TILE} DATA_DIR=${DATASET} ./run_model.sh
cd - 
