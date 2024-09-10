#!/bin/bash
set -e

echo "Setup PyTorch Test Enviroment for MaskRCNN Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/maskrcnn/inference/cpu/output/${PRECISION}"}
is_lkg_drop=$2
TEST_MODE=$3
DATASET_DIR=$4
CHECKPOINT_DIR=$5
MODE=$6
BATCH_SIZE=$7

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# Install dependency
cd models_v2/pytorch/maskrcnn/inference/cpu
MODEL_DIR=$(pwd)
./setup.sh

cd maskrcnn-benchmark
pip install -e .
pip install -r requirements.txt

cd ${MODEL_DIR}

# Run script
OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} DATASET_DIR=${DATASET_DIR} CHECKPOINT_DIR=${CHECKPOINT_DIR} MODE=${MODE} TEST_MODE=${TEST_MODE} BATCH_SIZE=${BATCH_SIZE} ./run_model.sh
cd -
