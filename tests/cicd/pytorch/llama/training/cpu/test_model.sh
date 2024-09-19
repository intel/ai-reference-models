#!/bin/bash
set -e

echo "Setup PyTorch Test Enviroment for Llama Training"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/llama/training/cpu/output/${PRECISION}"}
is_lkg_drop=$2
export DDP=$3

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

cd models_v2/pytorch/llama/training/cpu
MODEL_DIR=$(pwd)

# Prepare dataset
wget https://github.com/tloen/alpaca-lora/blob/main/alpaca_data.json
mv alpaca_data.json ${MODEL_DIR}

wget https://github.com/tloen/alpaca-lora/blob/main/templates/alpaca.json
mkdir ${MODEL_DIR}/templates
mv alpaca.json ${MODEL_DIR}/templates

# Install depedency
./setup.sh

# Run script
OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} TEST_MODE=${TEST_MODE} MODEL_DIR=${MODEL_DIR} ./run_model.sh
cd -
