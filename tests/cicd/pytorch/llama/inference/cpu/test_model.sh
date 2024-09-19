#!/bin/bash
set -e

echo "Setup PyTorch Test Enviroment for Llama Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/llama/inference/cpu/output/${PRECISION}"}
is_lkg_drop=$2
TEST_MODE=$3
BATCH_SIZE=$4

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

export INPUT_TOKEN=32
export OUTPUT_TOKEN=32
export FINETUNED_MODEL="meta-llama/Llama-2-7b-hf"
export BEAM_SIZE=4
CORE_PER_INSTANCE=$5

# Install dependency
cd models_v2/pytorch/llama/inference/cpu
MODEL_DIR=$(pwd)
./setup.sh

OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} FINETUNED_MODEL=${FINETUNED_MODEL} TEST_MODE=${TEST_MODE} MODEL_DIR=${MODEL_DIR} BATCH_SIZE=${BATCH_SIZE} ./run_model.sh 
cd -
