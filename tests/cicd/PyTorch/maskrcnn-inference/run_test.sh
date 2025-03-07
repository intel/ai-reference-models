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

echo "Setup PyTorch Test Enviroment for Mask R-CNN Inference"

PRECISION=$1
SCRIPT=$2
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/PyTorch/maskrcnn-inference/${SCRIPT}/${PRECISION}"}
WORKSPACE=$3
is_lkg_drop=$4
DATASET=$5
MODE='jit'

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD 
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# Install dependency:
./quickstart/object_detection/pytorch/maskrcnn/inference/cpu/setup.sh

# Install model:
cd models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/
python setup.py develop
cd - 

# Install pre-trained model:
export CHECKPOINT_DIR=$(pwd)/tests/cicd/output/PyTorch/maskrcnn-inference/${PRECISION}
bash quickstart/object_detection/pytorch/maskrcnn/inference/cpu/download_model.sh

# Run script
OUTPUT_DIR=${OUTPUT_DIR} DATASET_DIR=${DATASET} PRECISION=${PRECISION} CHECKPOINT_DIR=${CHECKPOINT_DIR} MODE=${MODE} ./quickstart/object_detection/pytorch/maskrcnn/inference/cpu/${SCRIPT}
