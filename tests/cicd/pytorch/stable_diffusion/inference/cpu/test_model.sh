#!/bin/bash
set -e

echo "Setup PyTorch Test Enviroment for Stable Diffusion Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/stable_diffusion/inference/cpu/output/${PRECISION}"}
is_lkg_drop=$2
TEST_MODE=$3
DATASET_DIR=$4
MODE=$5
DISTRIBUTED=$6

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# Install dependency
cd models_v2/pytorch/stable_diffusion/inference/cpu
MODEL_DIR=$(pwd)
./setup.sh

# Run script
OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} DATASET_DIR=${DATASET_DIR} MODE=${MODE} TEST_MODE=${TEST_MODE} DISTRIBUTED=${DISTRIBUTED} ./run_model.sh
cd -
