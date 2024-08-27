#!/bin/bash
set -e

echo "Setup PyTorch Test Enviroment for CHATGLMv3 Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/chatglm/inference/cpu/output/${PRECISION}"}
is_lkg_drop=$2
TEST_MODE=$3
BATCH_SIZE=$4

mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# Install dependency
cd models_v2/pytorch/chatglm/inference/cpu
./setup.sh

export REVISION=9addbe01105ca1939dd60a0e5866a1812be9daea
using BEAM_SIZE=4

INPUT_TOKEN=32
OUTPUT_TOKEN=32

OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} TEST_MODE=${TEST_MODE} BATCH_SIZE=${BATCH_SIZE} REVISION=${REVISION} BEAM_SIZE=${BEAM_SIZE} INPUT_TOKEN=${INPUT_TOKEN} OUTPUT_TOKEN=${OUTPUT_TOKEN} ./run_model.sh
cd -
