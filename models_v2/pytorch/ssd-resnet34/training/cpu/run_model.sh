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

MODEL_DIR=${MODEL_DIR:-$PWD}

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
else
    echo "Please set TEST_MODE to THROUGHPUT or ACCURACY"
    exit
fi

if [ ! -e "${MODEL_DIR}/train.py" ]; then
  echo "Could not find the script of train.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the train.py exist at."
  exit 1
fi

if [ -z "${CHECKPOINT_DIR}" ]; then
  echo "The pretrained model is not set"
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR is not set"
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
  echo "Precision is not set"
  exit 1
fi

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""
if [ "$PRECISION" == "bf16" ]; then
    ARGS="$ARGS --autocast"
    echo "### running bf16 datatype"
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    echo "### running fp32 datatype"
elif [[ "$PRECISION" == "bf32" ]]; then
    ARGS="$ARGS --bf32"
    echo "### running bf32 datatype"
else
    echo "The specified precision '$PRECISION' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf32, and bf16"
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

ARGS_IPEX=""

if [ "${TEST_MODE}" == "THROUGHPUT" ]; then
    echo "Running throughput training mode"
    ARGS_IPEX="$ARGS_IPEX --nodes-list 0"
    BATCH_SIZE=${BATCH_SIZE:-224}
    ARGS="$ARGS --epochs 70"
    ARGS="$ARGS --pretrained-backbone ${CHECKPOINT_DIR}/ssd/resnet34-333f7ec4.pth"
    ARGS="$ARGS --performance_only"
    ARGS="$ARGS -w 20"
    ARGS="$ARGS -iter 100"

    LOG=${OUTPUT_DIR}/train_ssdresnet34_${PRECISION}_throughput.log
    LOG_0=${OUTPUT_DIR}/train_ssdresnet34_${PRECISION}_throughput*
else
    echo "Running accuracy training mode"
    ARGS_IPEX="$ARGS_IPEX --ninstances 1"
    ARGS_IPEX="$ARGS_IPEX --ncore_per_instance ${CORES_PER_INSTANCE}"
    BATCH_SIZE=${BATCH_SIZE:-100}
    ARGS="$ARGS --epochs 5"
    ARGS="$ARGS --pretrained-backbone ${CHECKPOINT_DIR}/ssd/resnet34-333f7ec4.pth"

    LOG=${OUTPUT_DIR}/train_ssdresnet34_${PRECISION}_accuracy.log
    LOG_0=${OUTPUT_DIR}/train_ssdresnet34_${PRECISION}_accuracy*
fi

if [ "$DISTRIBUTED" ]; then
    oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
    source $oneccl_bindings_for_pytorch_path/env/setvars.sh
    ARGS_IPEX="$ARGS_IPEX --distributed"
    ARGS_IPEX="$ARGS_IPEX  --nnodes ${NNODES}"
    ARGS_IPEX="$ARGS_IPEX --hostfile ${HOSTFILE}"
    ARGS_IPEX="$ARGS_IPEX --logical_core_for_ccl --ccl_worker_count 8"
    ARGS="$ARGS --world_size ${NUM_RANKS}"
    ARGS="$ARGS --backend ccl"

    LOG= ${OUTPUT_DIR}/train_ssdresnet34_${PRECISION}_throughput_dist.log
    LOG_0= ${OUTPUT_DIR}/train_ssdresnet34_${PRECISION}_throughput_dist*
fi

rm -rf ${LOG_0}

python -m intel_extension_for_pytorch.cpu.launch \
    --memory-allocator jemalloc \
    ${ARGS_IPEX} \
    ${MODEL_DIR}/train.py \
    --warmup-factor 0 \
    --lr 2.5e-3 \
    --threshold=0.23 \
    --seed 2000 \
    --log-interval 10 \
    --data ${DATASET_DIR}/coco \
    --batch-size ${BATCH_SIZE} \
    $ARGS 2>&1 | tee ${LOG}

# For the summary of results
wait

throughput=$(grep 'Throughput:' ${LOG_0} |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
    BEGIN {
            sum = 0;
    i = 0;
        }
        {
            if ($1 != 0) {
                sum = sum + $1;
                i++;
            }
        }
    END   {
        if (i > 0) {
            sum = sum / i;
            printf("%.3f", sum);
        } else {
            print "No latency values found.";
        }

    }')
echo "--------------------------------Performance Summary per Numa Node--------------------------------"
echo ""SSD-RN34";"training throughput";$PRECISION; ${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
accuracy=$(grep 'Accuracy:' ${LOG_0} |sed -e 's/.*Accuracy//;s/[^0-9.]//g' |awk '
    BEGIN {
            sum = 0;
    i = 0;
        }
        {
            if ($1 != 0) {
                sum = sum + $1;
                i++;
            }
        }
    END   {
        if (i > 0) {
            sum = sum / i;
            printf("%.3f", sum);
        } else {
            print "No latency values found.";
        }

    }')

latency=$(grep 'train latency ' ${LOG_0} |sed -e 's/.*inference latency\s*//;s/[^0-9.]//g' |awk '
    BEGIN {
            sum = 0;
    i = 0;
        }
        {
            if ($1 != 0) {
                sum = sum + $1;
                i++;
            }
        }
    END   {
        if (i > 0) {
            sum = sum / i;
            printf("%.3f", sum);
        } else {
            print "No latency values found.";
        }

    }')
echo "--------------------------------Performance Summary per Numa Node--------------------------------"
echo ""SSD-RN34";"training latency";$PRECISION; ${BATCH_SIZE};${latency}" | tee -a ${OUTPUT_DIR}/summary.log
echo ""SSD-RN34";"throughput";"accuracy";"latency";$PRECISION; ${BATCH_SIZE};${throughput};${accuracy};${latency}" | tee -a ${OUTPUT_DIR}/summary.log

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: fps
- key: latency
  value: $latency
  unit: ms
- key: accuracy
  value: $accuracy
  unit: AP
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
