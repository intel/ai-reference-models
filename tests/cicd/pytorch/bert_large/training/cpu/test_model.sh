#!/bin/bash
set -e

echo "Setup PyTorch Test Enviroment for BERT LARGE Training"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/bert_large/training/cpu/output/${PRECISION}"}
is_lkg_drop=$2
DATASET_DIR=$3
DDP=$4

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# Install dependency
cd models_v2/pytorch/bert_large/training/cpu
./setup.sh

# Get CONFIG_FILE:
if [ -f "bert_config.json" ]; then
  echo "The eval data file exists. Skipping download."
else
  wget -O bert_config.json 'https://drive.google.com/uc?export=download&id=1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW'
fi
BERT_MODEL_CONFIG=$(pwd)/bert_config.json

if [ -d "CHECKPOINT_DIR" ]; then
  echo "Skipping creating checkpoint folder."
else
  mkdir -p checkpoint_dir
fi

# Run script
# Phase 1
OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} BERT_MODEL_CONFIG=${BERT_MODEL_CONFIG} DDP=${DDP} TRAINING_PHASE=1 CHECKPOINT_DIR=$(pwd)/checkpoint_dir DATASET_DIR=${DATASET_DIR} TRAIN_SCRIPT=$(pwd)/run_pretrain_mlperf.py ./run_model.sh
# Phase 2
OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} DDP=${DDP} TRAINING_PHASE=2 PRETRAINED_MODEL=$(pwd)/checkpoint_dir DATASET_DIR=${DATASET_DIR} TRAIN_SCRIPT=$(pwd)/run_pretrain_mlperf.py ./run_model.sh
cd -
