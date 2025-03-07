#!/bin/bash
set -e
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "Setup PyTorch Test Enviroment for BERT LARGE Training"

PRECISION=$1
SCRIPT=$2
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/PyTorch/bert-large-training/${SCRIPT}/${PRECISION}"}
WORKSPACE=$3
is_lkg_drop=$4
DATASET=$5

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD 
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# Install dependency
./quickstart/language_modeling/pytorch/bert_large/training/cpu/setup.sh

# For phase 1 get the bert config from [here](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT)
export BERT_MODEL_CONFIG=$current_directory/models/language_modeling/pytorch/bert_large/training/gpu/bert_config.json

export CHECKPOINT_DIR=$(pwd)/checkpoint_phase1_dir

# Run script
OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} CHECKPOINT_DIR=${CHECKPOINT_DIR} DATASET_DIR=${DATASET} BERT_MODEL_CONFIG=${BERT_MODEL_CONFIG} ./quickstart/language_modeling/pytorch/bert_large/training/cpu/run_bert_pretrain_phase1.sh

OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} PRETRAINED_MODEL=${CHECKPOINT_DIR} DATASET_DIR=${DATASET} ./quickstart/language_modeling/pytorch/bert_large/training/cpu/run_bert_pretrain_phase2.sh
