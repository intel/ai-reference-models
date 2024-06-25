#!/bin/bash
set -e

echo "Setup IPEX-XPU Test Enviroment for 3D Unet Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/3d_unet/inference/gpu/output/${PRECISION}"}
is_lkg_drop=$2
platform=$3
DUMMY=$4
DATASET_DIR=$5

if [[ "${platform}" == "flex=gpu" || "${platform}" == "ATS-M" ]]; then
    runner="Flex"
elif [[ "${platform}" == "max-gpu" || "${platform}" == "pvc" ]]; then
    exit 1
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
cd models_v2/pytorch/3d_unet/inference/gpu
python3 -m pip install -r requirements.txt
export PYTHONPATH=$(pwd)/../../../../common
./setup.sh

OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} DUMMY=${DUMMY} DATASET_DIR=${DATASET_DIR} ./run_model.sh
cd -
