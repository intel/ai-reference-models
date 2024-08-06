#!/usr/bin/env bash
#
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

if [ ! -e "${MODEL_DIR}/maskrcnn-benchmark/tools/train_net.py" ]; then
  echo "Could not find the script of train.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the train.py exist."
  exit 1
fi

if [ ! -d "${DATASET_DIR}/coco" ]; then
  echo "The DATASET_DIR \${DATASET_DIR}/coco does not exist"
  exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}
rm -rf ${OUTPUT_DIR}/summary.log
rm -rf ${OUTPUT_DIR}/results.yaml

if [ -z "${PRECISION}" ]; then
  echo "The PRECISION env variable is not set"
  exit 1
fi

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""

if [[ "$PRECISION" == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    echo "### running bf16 datatype"
elif [[ "$PRECISION" == "bf32" ]]; then
    ARGS="$ARGS --bf32"
    echo "### running bf32 datatype"
elif [[ "$PRECISION" == "fp32" || "$PRECISION" == "avx-fp32" ]]; then
    echo "### running fp32 datatype"
else
    echo "The specified precision '$PRECISION' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf16, and bf32."
    exit 1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

CORES_PER_INSTANCE=$CORES

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export USE_IPEX=1
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

export TRAIN=1

throughput="N/A"
accuracy="N/A"
latency="N/A"

if [ "$DISTRIBUTED" ]; then
    NNODES=${NNODES:-1}
    if [ -z "${HOSTFILE}" ]; then
      echo "The HOSTFILE env variable is not set"
      exit 1
    fi
    if [ -z "${LOCAL_BATCH_SIZE}" ]; then
      echo "The required environment variable LOCAL_BATCH_SIZE has not been set"
      exit 1
    fi
    NUM_RANKS=$(( NNODES * SOCKETS ))

    oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
    source $oneccl_bindings_for_pytorch_path/env/setvars.sh
    export FI_PROVIDER=psm3
    export PSM3_HAL=sockets

    rm -rf ${OUTPUT_DIR}/maskrcnn_dist_training_log_${PRECISION}*

    LOG= ${OUTPUT_DIR}/maskrcnn_dist_training_log_${PRECISION}.log
    LOG_0=${LOG}

    python -m intel_extension_for_pytorch.cpu.launch \
        --memory-allocator tcmalloc \
        --nnodes ${NNODES} \
        --hostfile ${HOSTFILE} \
        --logical-cores-for-ccl --ccl_worker_count 8 \
        ${MODEL_DIR}/maskrcnn-benchmark/tools/train_net.py \
        $ARGS \
        --iter-warmup 10 \
        -i 20 \
        -b ${LOCAL_BATCH_SIZE} \
        --config-file "${MODEL_DIR}/maskrcnn-benchmark/configs/e2e_mask_rcnn_R_50_FPN_1x_coco2017_tra.yaml" \
        --skip-test \
        --backend ccl \
        SOLVER.MAX_ITER 720000 \
        SOLVER.STEPS '"(60000, 80000)"' \
        SOLVER.BASE_LR 0.0025 \
        MODEL.DEVICE cpu \
        2>&1 | tee ${LOG}

    wait

else
    BATCH_SIZE=${BATCH_SIZE:-112}

    rm -rf ${OUTPUT_DIR}/maskrcnn_${PRECISION}_train_throughput*

    LOG=${OUTPUT_DIR}/maskrcnn_${PRECISION}_train_throughput.log
    LOG_0=${OUTPUT_DIR}/maskrcnn_${PRECISION}_train_throughput*

    python -m intel_extension_for_pytorch.cpu.launch \
        --memory-allocator jemalloc \
        --nodes-list=0 \
        ${MODEL_DIR}/maskrcnn-benchmark/tools/train_net.py \
        $ARGS \
        --iter-warmup 10 \
        -i 20 \
        -b ${BATCH_SIZE} \
        --config-file "${MODEL_DIR}/maskrcnn-benchmark/configs/e2e_mask_rcnn_R_50_FPN_1x_coco2017_tra.yaml" \
        --skip-test \
        SOLVER.MAX_ITER 720000 \
        SOLVER.STEPS '"(480000, 640000)"' \
        SOLVER.BASE_LR 0.0025 \
        MODEL.DEVICE cpu \
        2>&1 | tee ${LOG}

    wait
fi

if [ "$DISTRIBUTED" ]; then
    throughput=$(grep 'Training throughput:' ${LOG_0} |sed -e 's/.*Training throughput//;s/[^0-9.]//g' |awk '
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
    echo ""maskrcnn";"training distributed throughput";"latency";"accuracy";${PRECISION};${LOCAL_BATCH_SIZE};${throughput};${latency};${accuracy};" | tee -a ${OUTPUT_DIR}/summary.log
else
    throughput=$(grep 'Training throughput:' ${LOG_0} |sed -e 's/.Trainng throughput//;s/[^0-9.]//g')
    echo ""maskrcnn";"training throughput";"latency";"accuracy";$PRECISION;${BATCH_SIZE};${throughput};${latency};${accuracy};" | tee -a ${OUTPUT_DIR}/summary.log
fi

latency=$(grep 'Latency:' ${LOG_0} |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: fps
- key: latency
  value: $latency
  unit: ms
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
