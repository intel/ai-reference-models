#!/bin/bash
set -e

echo "Setup PyTorch Test Enviroment for GPTJ Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/gptj/inference/cpu/output/${PRECISION}"}
is_lkg_drop=$2
TEST_MODE=$3

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# Install dependency
cd models_v2/pytorch/gptj/inference/cpu
./setup.sh

MODEL_DIR=$(pwd)
export INPUT_TOKEN=32
export OUTPUT_TOKEN=32
export BEAM_SIZE=4

# Run script
OUTPUT_DIR=${OUTPUT_DIR} MODEL_DIR=${MODEL_DIR} PRECISION=${PRECISION} TEST_MODE=${TEST_MODE} ./run_model.sh
cd -
