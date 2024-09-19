#!/bin/bash
set -e

echo "Setup PyTorch Test Enviroment for RN-50 Training"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/resnet50/training/cpu/output/${PRECISION}"}
is_lkg_drop=$2
DATASET_DIR=$3
DISTRIBUTED=$4
TRAINING_EPOCHS=$5

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# Install dependency
cd models_v2/pytorch/resnet50/training/cpu
MODEL_DIR=$(pwd)

# Default Batch Size 256
BATCH_SIZE=256

OUTPUT_DIR=${OUTPUT_DIR} DATASET_DIR=${DATASET_DIR} PRECISION=${PRECISION} DISTRIBUTED=${DISTRIBUTED} TRAINING_EPOCHS=${TRAINING_EPOCHS} BATCH_SIZE=${BATCH_SIZE} ./run_model.sh
cd -
