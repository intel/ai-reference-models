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

echo "Setup PyTorch Test Enviroment for BERT LARGE Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/PyTorch/bert-large-inference/${PRECISION}"}
WORKSPACE=$3
is_lkg_drop=$4
DATASET=$5

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

#export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD 
#export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
#export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# Install dependency
cd models_v2/pytorch/bert_large/inference/cpu
./setup.sh

# Get EVAL_DATA_FILE:
if [ -d "dev-v1.1.json" ]; then
  echo "The eval data file exists. Skipping download."
else
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
fi
export EVAL_DATA_FILE=$(pwd)/dev-v1.1.json

# Get Pretrained model:
if [ -d "bert_squad_model" ]; then
  echo "The pretrained model exists. Skipping download."
else
  mkdir bert_squad_model
  wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
  wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
fi
export FINETUNED_MODEL=$(pwd)/bert_squad_model

# Run script
OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} FINETUNED_MODEL=${FINETUNED_MODEL} EVAL_DATA_FILE=${EVAL_DATA_FILE} ./run_model.sh
cd -
