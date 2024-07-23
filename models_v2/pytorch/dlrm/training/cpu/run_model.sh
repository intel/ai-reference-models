#!/bin/bash
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
#

MODEL_DIR=${MODEL_DIR-$PWD}
if [ ! -e "${MODEL_DIR}/../../common/dlrm_s_pytorch.py"  ]; then
    echo "Could not find the script of dlrm_s_pytorch.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the dlrm_s_pytorch.py exist at."
    exit 1
fi
MODEL_SCRIPT=${MODEL_DIR}/../../common/dlrm_s_pytorch.py

echo "PRECISION: ${PRECISION}"
echo "DATASET_DIR: ${DATASET_DIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

mkdir -p ${OUTPUT_DIR}

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "Please set PRECISION: fp32, bf16, bf32"
  exit 1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`

CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | awk '{print $4}'`
SOCKETS=`lscpu | grep "Socket(s)" | awk '{print $2}'`
NUMA_NODES=`lscpu | grep "NUMA node(s)" | awk '{print $3}'`
NUMA_NODES_PER_SOCKETS=`expr $NUMA_NODES / $SOCKETS`
CORES_PER_NUMA_NODE=`expr $CORES_PER_SOCKET / $NUMA_NODES_PER_SOCKETS`

export OMP_NUM_THREADS=$CORES_PER_NUMA_NODE
LOG=${OUTPUT_DIR}/dlrm_training_log/${PRECISION}

BATCH_SIZE=${BATCH_SIZE:-32768}

if [ "$DISTRIBUTED"]; then
    BATCH_SIZE=${BATCH_SIZE:-32768}
    NUM_CCL_WORKER=${NUM_CCL_WORKER:-8}
    HOSTFILE=${HOSTFILE:-hostfile1}
    seed_num=1665468325 #1665462256 #$(date +%s)
    oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
    source $oneccl_bindings_for_pytorch_path/env/setvars.sh
    export TORCH_CPP_LOG_LEVEL=INFO
    export TORCH_DISTRIBUTED_DEBUG=INFO
    export CCL_LOG_LEVEL=info
    export CCL_ALLREDUCE=rabenseifner
    LOG=${OUTPUT_DIR}/dlrm_distribute_training_log/${PRECISION}
fi

rm -rf ${LOG}
mkdir -p ${LOG}

LOG_0="${LOG}/socket.log"
TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}

if [ "$DISTRIBUTED" ]; then
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        python -m intel_extension_for_pytorch.cpu.launch --enable_tcmalloc --logical_core_for_ccl --ccl_worker_count $NUM_CCL_WORKER --distributed --hostfile $HOSTFILE --nnodes $NODE \
        $MODEL_SCRIPT \
        --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz \
        --data-set=terabyte \
        --memory-map --mlperf-bin-loader --mlperf-bin-shuffle --round-targets=True --learning-rate=18.0 \
        --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
        --arch-sparse-feature-size=128 --max-ind-range=40000000 \
        --numpy-rand-seed=${seed_num} --print-auc --mlperf-auc-threshold=0.8025 \
        --lr-num-warmup-steps=8000   --lr-decay-start-step=70000 --lr-num-decay-steps=30000\
        --local-batch-size=${LOCAL_BATCH_SIZE} --print-freq=100 --print-time --ipex-interaction \
        --test-mini-batch-size=65536 --ipex-merged-emb --should-test --test-freq 6400\
        $ARGS |tee $LOG_0
    else
        export TORCHINDUCTOR_FREEZING=1
        python -m intel_extension_for_pytorch.cpu.launch --enable_tcmalloc --logical_core_for_ccl --ccl_worker_count $NUM_CCL_WORKER --distributed --hostfile $HOSTFILE --nnodes $NODE \
        $MODEL_SCRIPT \
        --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz \
        --data-set=terabyte \
        --memory-map --mlperf-bin-loader --mlperf-bin-shuffle --round-targets=True --learning-rate=18.0 \
        --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
        --arch-sparse-feature-size=128 --max-ind-range=40000000 \
        --numpy-rand-seed=${seed_num} --print-auc --mlperf-auc-threshold=0.8025 \
        --lr-num-warmup-steps=8000   --lr-decay-start-step=70000 --lr-num-decay-steps=30000\
        --local-batch-size=${LOCAL_BATCH_SIZE} --print-freq=100 --print-time \
        --test-mini-batch-size=65536 --should-test --test-freq 6400 --inductor\
        $ARGS |tee $LOG_0
    fi
else
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        python -m intel_extension_for_pytorch.cpu.launch --node_id=0 --enable_tcmalloc $MODEL_SCRIPT \
        --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz \
        --data-set=terabyte \
        --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
        --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
        --arch-sparse-feature-size=128 --max-ind-range=40000000 \
        --numpy-rand-seed=727 --print-auc --mlperf-auc-threshold=0.8025 \
        --mini-batch-size=${BATCH_SIZE} --print-freq=100 --print-time --ipex-interaction \
        --test-mini-batch-size=16384 --ipex-merged-emb \
        $ARGS |tee $LOG_0
    else
        export TORCHINDUCTOR_FREEZING=1
        python -m intel_extension_for_pytorch.cpu.launch --node_id=0 --enable_tcmalloc $MODEL_SCRIPT \
        --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz \
        --data-set=terabyte \
        --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
        --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
        --arch-sparse-feature-size=128 --max-ind-range=40000000 \
        --numpy-rand-seed=727 --print-auc --mlperf-auc-threshold=0.8025 \
        --mini-batch-size=${BATCH_SIZE} --print-freq=100 --print-time \
        --test-mini-batch-size=16384 --inductor \
        $ARGS |tee $LOG_0
    fi
fi

throughput=$(echo "$LOG_0" | grep -oP 'Throughput:  \K[^ ]+')
accuracy=$(echo "$LOG_0" | grep -oP 'accuracy \K[^ ]+')
latency=$(echo "$LOG_0" | grep -oP 'latency:  \K[^ ]+')

echo "Throughput: $throughput"
echo "Accuracy: $accuracy"
echo "Latency: $latency"

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: samples per second
- key: latency
  value: $latency
  unit: s
- key: accuracy
  value: $accuracy
  unit: percentage
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
