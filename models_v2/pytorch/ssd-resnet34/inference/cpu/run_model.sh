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

if [ ! -e "${MODEL_DIR}/infer.py" ]; then
  echo "Could not find the script of infer.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the infer.py exist"
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
  echo "The OUTPUT_DIR is not set"
  exit 1
fi

mkdir -p ${OUTPUT_DIR}

if [ -z "${PRECISION}" ]; then
  echo "PRECISION is not set"
  exit 1
fi

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""
if [[ "$PRECISION" == "int8" || "$PRECISION" == "avx-int8" ]]; then
    ARGS="$ARGS --int8"
    ARGS="$ARGS --seed 1 --threshold 0.2 --configure ${MODEL_DIR}/pytorch_default_recipe_ssd_configure.json"
    export DNNL_GRAPH_CONSTANT_CACHE=1
    echo "### running int8 datatype"
elif [[ "$PRECISION" == "bf16" ]]; then
    ARGS="$ARGS --autocast"
    echo "### running bf16 datatype"
elif [[ "$PRECISION" == "fp32" || "$PRECISION" == "avx-fp32" ]]; then
    echo "### running fp32 datatype"
elif [[ "$PRECISION" == "bf32" ]]; then
    ARGS="$ARGS --bf32"
    echo "### running bf32 datatype"
else
    echo "The specified precision '$PRECISION' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf16, int8, bf32, and avx-int8"
    exit 1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export USE_IPEX=1
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0


if [ "$THROUGHPUT" ]; then
    rm -rf ${OUTPUT_DIR}/throughput_log*
else
    rm -rf ${OUTPUT_DIR}/ssdresnet34_${PRECISION}_inference_accuracy*
fi

ARGS_IPEX=""

if [ -z "$THROUGHPUT" ]; then
    echo "Running accuracy mode"
    ARGS="$ARGS --accuracy-mode"
    BATCH_SIZE=16
    BATCH_PER_STREAM=1
else
    echo "Running throughput mode"
    ARGS_IPEX="$ARGS_IPEX --throughput_mode"
    ARGS="$ARGS --iteration 50"
    ARGS="$ARGS -w 10"
    ARGS="$ARGS --throughput-mode"
    BATCH_SIZE=${BATCH_SIZE:-112}
    BATCH_PER_STREAM=2

fi


if [ "${WEIGHT_SHARING}" ]; then
    echo "### Running the test with runtime extension."
    ARGS="$ARGS --async-execution"
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    CORES_PER_INSTANCE=$CORES
    INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
    LAST_INSTANCE=`expr $INSTANCES - 1`
    INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`

    CORES_PER_STREAM=1
    STREAM_PER_INSTANCE=`expr $CORES / $CORES_PER_STREAM`
    BATCH_SIZE=`expr $BATCH_PER_STREAM \* $STREAM_PER_INSTANCE`
    export OMP_NUM_THREADS=$CORES_PER_STREAM

    for i in $(seq 0 $LAST_INSTANCE); do
        numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
        start_core_i=`expr $i \* $CORES_PER_INSTANCE`
        end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`

        if [ "$THROUGHPUT" ]; then
            LOG_i=throughput_log_ssdresnet34_weight_sharing_${i}.log
        else
            LOG_i=accuracy_log_ssdresnet34_weight_sharing_${i}.log
        fi
        echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
        numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python -u \
            ${MODEL_DIR}/infer_weight_sharing.py \
            --data ${DATASET_DIR}/coco \
            --device 0 \
            --checkpoint ${CHECKPOINT_DIR}/pretrained/resnet34-ssd1200.pth \
            -j 0 \
            --no-cuda \
            --batch-size ${BATCH_SIZE} \
            --jit \
            --number-instance $STREAM_PER_INSTANCE \
            --use-multi-stream-module \
            --instance-number $i \
            $ARGS 2>&1 | tee ${OUTPUT_DIR}/$LOG_i &
    done
    wait
else
    if [ "$THROUGHPUT" ]; then
        LOG=${OUTPUT_DIR}/throughput_log_ssdresnet34_${PRECISION}.log
    else
        LOG=${OUTPUT_DIR}/accuracy_log_ssdresnet34_${PRECISION}.log
    fi
    python -m intel_extension_for_pytorch.cpu.launch \
        ${ARGS_IPEX}  \
        ${MODEL_DIR}/infer.py \
        --data ${DATASET_DIR}/coco \
        --device 0 \
        --checkpoint ${CHECKPOINT_DIR}/pretrained/resnet34-ssd1200.pth \
        -j 0 \
        --no-cuda \
        --batch-size ${BATCH_SIZE} \
        --jit \
        $ARGS 2>&1 | tee ${LOG}

    wait

fi

# post-processing
throughput="0"
accuracy="0"
latency="0"

if [ "$THROUGHPUT" ]; then
    LOG=${OUTPUT_DIR}/throughput_log_ssdresnet34*
else
    LOG=${OUTPUT_DIR}/accuracy_log_ssdresnet34*
fi

echo $LOG

throughput=$(grep 'Throughput:' ${LOG} |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
            print "No throughput values found.";
        }

    }')
accuracy=$(grep 'Accuracy:' ${LOG} |sed -e 's/.*Accuracy//;s/[^0-9.]//g' |awk '
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
            print "No accuracy values found.";
        }

    }')
latency=$(grep 'inference latency ' ${LOG} |sed -e 's/.*inference latency\s*//;s/[^0-9.]//g' |awk '
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
