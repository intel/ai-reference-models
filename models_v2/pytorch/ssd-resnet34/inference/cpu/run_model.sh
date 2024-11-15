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

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    echo "TEST_MODE set to REALTIME"
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
else
    echo "Please set TEST_MODE to THROUGHPUT, REALTIME or ACCURACY"
    exit
fi

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
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}
rm -rf ${OUTPUT_DIR}/summary.log
rm -rf ${OUTPUT_DIR}/results.yaml

if [ -z "${PRECISION}" ]; then
  echo "PRECISION is not set"
  exit 1
fi

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""
if [[ "$PRECISION" == "int8" || "$PRECISION" == "avx-int8" ]]; then
    if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
        NUMA_NODES=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
        CORES_PER_NODE=`expr $TOTAL_CORES / $NUMA_NODES`
        BATCH_SIZE=${BATCH_SIZE:-`expr $CORES_PER_NODE \* 2`}
    fi
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

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    rm -rf ${OUTPUT_DIR}/ssdresnet34_${PRECISION}_inference_throughput*
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    BATCH_SIZE=${BATCH_SIZE:-112}
    mode=throughput
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    BATCH_SIZE=${BATCH_SIZE:- 1}
    rm -rf ${OUTPUT_DIR}/ssdresnet34_${PRECISION}_inference_latency*
    CORES=`lscpu | grep Core | awk '{print $4}'`
    CORES_PER_INSTANCE=4
    INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET=`expr $CORES / $CORES_PER_INSTANCE`
    mode=latency
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    rm -rf ${OUTPUT_DIR}/ssdresnet34_${PRECISION}_inference_accuracy*
    mode=accuracy
fi

weight_sharing=true
if [ -z "${WEIGHT_SHARING}" ]; then
  weight_sharing=false
else
  echo "### Running the test with runtime extension."
  weight_sharing=true
fi

if [ "$weight_sharing" = true ]; then
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    if [[ "$TEST_MODE" == "THROUGHPUT" || "$TEST_MODE" == "ACCURACY" ]]; then
        async=true
        if [ "$async" = true ]; then
            ARGS="$ARGS --async-execution"
        fi
        CORES=`lscpu | grep Core | awk '{print $4}'`
        TOTAL_CORES=`expr $CORES \* $SOCKETS`
        CORES_PER_INSTANCE=$CORES
        INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
        LAST_INSTANCE=`expr $INSTANCES - 1`
        INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
        CORES_PER_STREAM=1
        STREAM_PER_INSTANCE=`expr $CORES / $CORES_PER_STREAM`
        export OMP_NUM_THREADS=$CORES_PER_STREAM
    fi

    if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
        BATCH_PER_STREAM=2
        BATCH_SIZE=`expr $BATCH_PER_STREAM \* $STREAM_PER_INSTANCE`
        for i in $(seq 0 $LAST_INSTANCE); do
            numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
            start_core_i=`expr $i \* $CORES_PER_INSTANCE`
            end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
            LOG_i=ssdresnet34_${PRECISION}_inference_throughput_log_weight_sharing_${i}.log

            echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
            numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python -u \
                ${MODEL_DIR}/infer_weight_sharing.py \
                --data ${DATASET_DIR}/coco \
                --device 0 \
                --checkpoint ${CHECKPOINT_DIR}/pretrained/resnet34-ssd1200.pth \
                -w 10 \
                -j 0 \
                --no-cuda \
                --iteration 50 \
                --batch-size ${BATCH_SIZE} \
                --jit \
                --number-instance $STREAM_PER_INSTANCE \
                --use-multi-stream-module \
                --instance-number $i \
                $ARGS 2>&1 | tee ${OUTPUT_DIR}/$LOG_i &
        done
        wait
    elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
        BATCH_PER_STREAM=1
        BATCH_SIZE=${BATCH_SIZE:- `expr $BATCH_PER_STREAM \* $STREAM_PER_INSTANCE`}
        numa_node_i=0
        start_core_i=0
        end_core_i=`expr 0 + $CORES_PER_INSTANCE - 1`

        echo "### running on instance 0, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
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
            --instance-number 0 \
            --accuracy-mode \
            $ARGS 2>&1 | tee ${OUTPUT_DIR}/ssdresnet34_${PRECISION}_inference_accuracy.log
        wait
    elif [[ "$TEST_MODE" == "REALTIME" ]]; then
        SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
        export OMP_NUM_THREADS=$CORES_PER_INSTANCE

        python -m intel_extension_for_pytorch.cpu.launch \
            --memory-allocator jemalloc \
            --ninstance ${SOCKETS} \
            ${MODEL_DIR}/infer_weight_sharing.py \
            --data ${DATASET_DIR}/coco \
            --device 0 \
            --checkpoint ${CHECKPOINT_DIR}/pretrained/resnet34-ssd1200.pth \
            -w 20 \
            -j 0 \
            --no-cuda \
            --iteration 200 \
            --batch-size ${BATCH_SIZE} \
            --jit \
            --number-instance $INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET \
            $ARGS 2>&1 | tee ${OUTPUT_DIR}/ssdresnet34_${PRECISION}_inference_latency.log
        wait
    fi
else
    if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
        BATCH_SIZE=${BATCH_SIZE:-2}
        python -m intel_extension_for_pytorch.cpu.launch \
            --throughput_mode \
            ${MODEL_DIR}/infer.py \
            --data ${DATASET_DIR}/coco \
            --device 0 \
            --checkpoint ${CHECKPOINT_DIR}/pretrained/resnet34-ssd1200.pth \
            -w 10 \
            -j 0 \
            --no-cuda \
            --iteration 50 \
            --batch-size ${BATCH_SIZE} \
            --jit \
            --throughput-mode \
            $ARGS 2>&1 | tee ${OUTPUT_DIR}/ssdresnet34_${PRECISION}_inference_throughput.log
        wait
    elif [[ "$TEST_MODE" == "REALTIME" ]]; then
        python -m intel_extension_for_pytorch.cpu.launch \
            --memory-allocator jemalloc \
            --latency_mode \
            ${MODEL_DIR}/infer.py \
            --data ${DATASET_DIR}/coco \
            --device 0 \
            --checkpoint ${CHECKPOINT_DIR}/pretrained/resnet34-ssd1200.pth \
            -w 20 \
            -j 0 \
            --no-cuda \
            --iteration 200 \
            --batch-size ${BATCH_SIZE} \
            --jit \
            --latency-mode \
            $ARGS 2>&1 | tee ${OUTPUT_DIR}/ssdresnet34_${PRECISION}_inference_${mode}.log
        wait
    elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
        BATCH_SIZE=${BATCH_SIZE:- 16}
        LOG_0=ssdresnet34_${PRECISION}_inference_accuracy.log
        python -m intel_extension_for_pytorch.cpu.launch --log-dir ${OUTPUT_DIR} \
            ${MODEL_DIR}/infer.py \
            --data ${DATASET_DIR}/coco \
            --device 0 \
            --checkpoint ${CHECKPOINT_DIR}/pretrained/resnet34-ssd1200.pth \
            -j 0 \
            --no-cuda \
            --batch-size ${BATCH_SIZE} \
            --jit \
            --accuracy-mode \
            $ARGS 2>&1 | tee ${OUTPUT_DIR}/$LOG_0
        wait
    fi
fi

# post-processing
throughput="N/A"
accuracy="N/A"
latency="N/A"

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    LOG=${OUTPUT_DIR}/throughput_log_ssdresnet34*
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    LOG=${OUTPUT_DIR}/latency_log_ssdresnet34*
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    LOG=${OUTPUT_DIR}/accuracy_log_ssdresnet34*
fi

echo $LOG
if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/ssdresnet34_${PRECISION}_inference_${mode}* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
    echo "--------------------------------Performance Summary per Numa Node--------------------------------"
    echo ""SSD-RN34";"throughput";$PRECISION; ${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
if [[ "$TEST_MODE" == "REALTIME" ]]; then
    latency=$(grep 'P99 Latency' ${OUTPUT_DIR}/ssdresnet34_${PRECISION}_inference_${mode}* |sed -e 's/.*P99 Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET '
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
        printf("%.3f ms", sum);
    }')
    echo "--------------------------------Performance Summary per Socket--------------------------------"
    echo ""SSD-RN34";"p99_latency";$PRECISION; ${BATCH_SIZE};${latency}" | tee -a ${OUTPUT_DIR}/summary.log
    latency=$(grep 'inference latency:' ${OUTPUT_DIR}/ssdresnet34_${PRECISION}_inference_${mode}* |sed -e 's/.*inference latency//;s/[^0-9.]//g' |awk '
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
    echo "--------------------------------Performance Summary per Numa Node--------------------------------"
    echo ""SSD-RN34";"latency";$PRECISION; ${BATCH_SIZE};${latency}" | tee -a ${OUTPUT_DIR}/summary.log
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    accuracy=$(grep 'Accuracy:' ${OUTPUT_DIR}/ssdresnet34_${PRECISION}_inference_${mode}* |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
    echo ""SSD-RN34";"accuracy";$PRECISION; ${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
fi

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
