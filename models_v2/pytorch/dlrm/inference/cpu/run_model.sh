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

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
else
    echo "Please set TEST_MODE to THROUGHPUT or ACCURACY"
    exit
fi

MODEL_DIR=${MODEL_DIR-$PWD}
if [ ! -e "${MODEL_DIR}/../../common/dlrm_s_pytorch.py"  ]; then
    echo "Could not find the script of dlrm_s_pytorch.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the dlrm_s_pytorch.py exist at."
    exit 1
fi
MODEL_SCRIPT=${MODEL_DIR}/../../common/dlrm_s_pytorch.py

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "Please set PRECISION: int8, fp32, bf16, bf32"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}
if [ $THROUGHPUT ]; then
    LOG=${OUTPUT_DIR}/dlrm_inference_performance_log/${PRECISION}
else
    LOG=${OUTPUT_DIR}/dlrm_inference_accuracy_log/${PRECISION}
fi

rm -rf ${LOG}
mkdir -p ${LOG}
rm -rf ${OUTPUT_DIR}/summary.log
rm -rf ${OUTPUT_DIR}/results.yaml

CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | awk '{print $4}'`
SOCKETS=`lscpu | grep "Socket(s)" | awk '{print $2}'`
NUMA_NODES=`lscpu | grep "NUMA node(s)" | awk '{print $3}'`
NUMA_NODES_PER_SOCKETS=`expr $NUMA_NODES / $SOCKETS`
CORES_PER_NUMA_NODE=`expr $CORES_PER_SOCKET / $NUMA_NODES_PER_SOCKETS`

# Runs with default value when BATCH SIZE is not set:
BATCH_SIZE=${BATCH_SIZE:-128}

ARGS=""
if [[ $PRECISION == "int8" ]]; then
    echo "running int8 path"
    ARGS="$ARGS --num-cpu-cores=$CORES_PER_NUMA_NODE --int8 --int8-configure=${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/int8_configure.json"
elif [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    echo "running bf16 path"
elif [[ $PRECISION == "fp32" ]]; then
    echo "running fp32 path"
elif [[ $PRECISION == "bf32" ]]; then
    echo "running bf32 path"
    ARGS="$ARGS --bf32"
else
    echo "The specified PRECISION '${PRECISION}' is unsupported."
    echo "Supported PRECISIONs are: fp32, bf32, bf16, and int8"
    exit 1
fi

export OMP_NUM_THREADS=$CORES_PER_SOCKET
if [ "$TEST_MODE" == "THROUGHPUT" ]; then
    LOG="${LOG}/throughput.log"
else
    LOG="${LOG}/accuracy.log"
fi

TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}

if [ "$TEST_MODE" == "THROUGHPUT" ]; then
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        python -m intel_extension_for_pytorch.cpu.launch --throughput_mode --memory-allocator tcmalloc $MODEL_SCRIPT \
            --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz \
            --data-set=terabyte \
            --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
            --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
            --arch-sparse-feature-size=128 --max-ind-range=40000000 --ipex-interaction \
            --numpy-rand-seed=727  --inference-only --num-batches=1000 \
            --print-freq=10 --print-time --test-mini-batch-size=${BATCH_SIZE} --share-weight-instance=$CORES_PER_NUMA_NODE \
            $ARGS |tee $LOG
    else
        echo "### running with torch.compile inductor backend"
        export TORCHINDUCTOR_FREEZING=1
        python -m intel_extension_for_pytorch.cpu.launch --throughput_mode --memory-allocator tcmalloc $MODEL_SCRIPT \
            --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz \
            --data-set=terabyte \
            --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
            --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
            --arch-sparse-feature-size=128 --max-ind-range=40000000 --inductor \
            --numpy-rand-seed=727  --inference-only --num-batches=1000 \
            --print-freq=10 --print-time --test-mini-batch-size=${BATCH_SIZE} --share-weight-instance=$CORES_PER_NUMA_NODE \
            $ARGS |tee $LOG
    fi
else
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        python -m intel_extension_for_pytorch.cpu.launch --nodes-list=0 --memory-allocator tcmalloc $MODEL_SCRIPT \
        --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz \
        --data-set=terabyte \
        --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
        --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
        --arch-sparse-feature-size=128 --max-ind-range=40000000 \
        --numpy-rand-seed=727  --inference-only --ipex-interaction \
        --print-freq=100 --print-time --mini-batch-size=2048 --test-mini-batch-size=16384 \
        --test-freq=2048 --print-auc $ARGS \
        --load-model=${WEIGHT_PATH} | tee $LOG
    else
    echo "### running with torch.compile inductor backend"
    export TORCHINDUCTOR_FREEZING=1
    python -m intel_extension_for_pytorch.cpu.launch --nodes-list=0 --memory-allocator tcmalloc $MODEL_SCRIPT \
        --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz \
        --data-set=terabyte \
        --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
        --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
        --arch-sparse-feature-size=128 --max-ind-range=40000000 \
        --numpy-rand-seed=727  --inference-only --inductor \
        --print-freq=100 --print-time --mini-batch-size=2048 --test-mini-batch-size=16384 \
        --test-freq=2048 --print-auc $ARGS \
        --load-model=${WEIGHT_PATH} | tee $LOG
    fi
fi

throughput="N/A"
accuracy="N/A"
latency="N/A"

if [ "$TEST_MODE" == "THROUGHPUT" ]; then
  throughput=$(grep 'Throughput:' ${LOG} |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
  BEGIN {
          sum = 0;
          i = 0;
        }
        {
          sum = sum + $1;
          i++;
        }
  END   {
  sum = sum / i;
          printf("%.3f", sum);
  }')
  echo "--------------------------------Performance Summary per NUMA Node--------------------------------"
  echo ""dlrm";"throughput";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
else
  accuracy=$(grep 'Accuracy:' $LOG |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
  echo ""dlrm";"auc";${PRECISION};16384;${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
fi

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
