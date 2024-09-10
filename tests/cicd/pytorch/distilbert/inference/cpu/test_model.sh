#!/bin/bash
set -e

echo "Setup PyTorch Test Enviroment for DistilBERT Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/distilbert/inference/cpu/output/${PRECISION}"}
is_lkg_drop=$2
TEST_MODE=$3
DATASET_DIR=$4

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# Install dependency
cd models_v2/pytorch/distilbert/inference/cpu
./setup.sh

git clone https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
FINETUNED_MODEL=$(pwd)/distilbert-base-uncased-finetuned-sst-2-english

SEQUENCE_LENGTH=128
CORE_PER_INSTANCE=4
HF_DATASETS_OFFLINE=0

OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} DATASET_DIR=${DATASET_DIR} FINETUNED_MODEL=${FINETUNED_MODEL} TEST_MODE=${TEST_MODE} SEQUENCE_LENGTH=${SEQUENCE_LENGTH} CORE_PER_INSTANCE=${CORE_PER_INSTANCE} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE} ./run_model.sh
cd -
