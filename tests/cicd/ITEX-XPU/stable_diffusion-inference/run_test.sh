#!/bin/bash
set -e

echo "Setup ITEX-XPU Test Enviroment for Stable Diffusion Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/ITEX-XPU/stable_diffusion-inference/${PRECISION}"}
is_lkg_drop=$2

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/tensorflow_setup/bin/activate tensorflow
fi

# run following script
cd models_v2/tensorflow/stable_diffusion/inference/gpu
./setup.sh

OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} ./run_model.sh
cd - 
