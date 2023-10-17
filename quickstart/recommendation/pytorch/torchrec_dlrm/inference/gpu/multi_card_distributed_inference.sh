#!/usr/bin/env bash
# Copyright (c) 2023 Intel Corporation
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
#

MODEL_DIR=${MODEL_DIR-$PWD}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE-65536}
TOTAL_SAMPLES=${TOTAL_SAMPLES-4195197692}
NUM_NODES=${NUM_NODES-1}
SHARDING_PLAN=${SHARDING_PLAN-round_robin}

if [[ -z "$PRETRAINED_MODEL" ]]; then
    echo "The required environment variable $PRETRAINED_MODEL has not been set"
    exit 1
fi 

if [[ -z $PRECISION ]]; then
  echo "The required environment variable PRECISION has not been set"
  exit 1
fi

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [[ -z "${DATASET_DIR}" ]]; then
    echo "The required environment variable DATASET_DIR has not been set"
    exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "${DATASET_DIR} does not exist."
    exit 1
fi

# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

if [[ -z $NUM_OAM ]]; then
  echo "The required environment variable NUM_OAM has not been set."
  exit 1
fi

source /opt/intel/oneapi/setvars.sh
export ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1
export MASTER_ADDR='127.0.0.1'

echo "Running DLRM single-node multi-card inference on ${NUM_OAM} OAM Modules"

ARGS+=" --embedding_dim 128"
ARGS+=" --dense_arch_layer_sizes 512,256,128"
ARGS+=" --over_arch_layer_sizes 1024,1024,512,256,1"
ARGS+=" --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36"
ARGS+=" --validation_freq_within_epoch $((TOTAL_SAMPLES / (GLOBAL_BATCH_SIZE * 20 * 1000)))"
ARGS+=" --synthetic_multi_hot_criteo_path $DATASET_DIR"
ARGS+=" --multi_hot_sizes 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1"
#ARGS+=" --multi_hot_distribution_type uniform"
ARGS+=" --use_xpu"
ARGS+=" --epochs 1"
ARGS+=" --pin_memory"
ARGS+=" --mmap_mode"
ARGS+=" --batch_size $GLOBAL_BATCH_SIZE"
ARGS+=" --interaction_type=dcn"
ARGS+=" --dcn_num_layers=3"
ARGS+=" --adagrad"
ARGS+=" --dcn_low_rank_dim=512"
ARGS+=" --numpy_rand_seed=12345"
ARGS+=" --log_freq 10"
ARGS+=" --amp"
ARGS+=" --inference_only"
ARGS+=" --snapshot_dir ${PRETRAINED_MODEL}"
ARGS+=" --num_batches 50"
ARGS+=" --sharding_plan ${SHARDING_PLAN}"
ARGS+=" --num_nodes ${NUM_NODES}"
ARGS+=" --learning_rate 0.005"

if [[ "$PRECISION" == "fp16" ]];then
    ARGS+=" --fp16"
    echo $ARGS 
else
    echo "DLRM Inference workload currently supports fp16 precision"
    exit 1
fi

if [[ "${NUM_OAM}" == "4" ]]; then
    np=8
    ppn=$np
    ZE_AFFINITY_MASK="0.0,0.1,1.0,1.1,2.0,2.1,3.0,3.1"
else
    echo "Currently only x4 OAM Modules are supported."
    exit 1
fi 

cd ${MODEL_DIR}/models/recommendation/pytorch/torchrec_dlrm/inference/gpu
I_MPI_DEBUG=6 ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK mpirun -np $np -ppn $ppn --prepend-rank python -u dlrm_main.py ${ARGS} 2>&1 | tee ${OUTPUT_DIR}/dlrm_x${NUM_OAM}_inference.log
