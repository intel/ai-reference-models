#!/bin/bash
set -e

echo "Setup IPEX-XPU Test Enviroment for Stable Diffusion Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/stable_diffusion/inference/gpu/output/${PRECISION}"}
is_lkg_drop=$2
platform=$3
MULTI_TILE=$4

if [[ "${platform}" == "flex=gpu" || "${platform}" == "ATS-M" ]]; then
    runner="Flex"
    multi_tile="False"
elif [[ "${platform}" == "max-gpu" || "${platform}" == "pvc" ]]; then
    runner="Max"
    multi_tile=${MULTI_TILE}
elif [[ "${platform}" == "arc" ]]; then
    runner="Arc"
    multi_tile="False"
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
cd models_v2/pytorch/stable_diffusion/inference/gpu

./setup.sh

OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} MULTI_TILE=${multi_tile} PLATFORM=${runner} ./run_model.sh
cd -
