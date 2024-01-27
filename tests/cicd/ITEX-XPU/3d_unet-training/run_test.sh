#!/bin/bash
set -e

echo "Setup ITEX-XPU Test Enviroment for 3D Unet Training"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/ITEX-XPU/3d_unet-training/${PRECISION}"}
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
cd models_v2/tensorflow/3d_unet/training/gpu
./setup.sh
pip uninstall horovod 
pip install intel-optimization-for-horovod

OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} MULTI_TILE=${MULTI_TILE} DATASET_DIR=${DATASET} ./run_model.sh
cd - 
