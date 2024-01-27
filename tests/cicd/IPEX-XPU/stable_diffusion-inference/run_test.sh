#!/bin/bash
set -e

echo "Setup IPEX Test Enviroment for Stable Diffusion Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/IPEX-XPU/stable_diffusion-inference/${PRECISION}"}
is_lkg_drop=$2
MULTI_TILE=$3
PLATFORM=$4

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
cd models_v2/pytorch/stable_diffusion/inference/gpu
./setup.sh

# Run script
OUTPUT_DIR=${OUTPUT_DIR} MULTI_TILE=${MULTI_TILE} PRECISION=${PRECISION} PLATFORM=${PLATFORM} ./run_model.sh
cd -
