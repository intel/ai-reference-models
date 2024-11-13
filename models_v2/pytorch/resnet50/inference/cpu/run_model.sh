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

ARGS=""
ARGS_IPEX=""

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
    LOG_PREFIX="resnet50_throughput_log"
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    echo "TEST_MODE set to REALTIME"
    LOG_PREFIX="resnet50_realtime_log"
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    LOG_PREFIX="resnet50_accuracy_log"
else
    echo "Please set TEST_MODE to THROUGHPUT or REALTIME or ACCURACY"
    exit
fi

MODEL_DIR=${MODEL_DIR-$PWD}
if [ ! -e "${MODEL_DIR}/../../common/main.py"  ]; then
    echo "Could not find the script of main.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the main.py exist at the: \${MODEL_DIR}/../../common/main.py"
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
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, avx-fp32, int8, bf32, avx-int8, or bf16."
  exit 1
fi

rm -rf "${OUTPUT_DIR}/${LOG_PREFIX}_*"

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi


if [[ $PRECISION == "int8" || $PRECISION == "avx-int8" ]]; then
    echo "running int8 path"
    ARGS="$ARGS --int8 --configure-dir ${MODEL_DIR}/../../common/resnet50_configure_sym.json"
elif [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16 --jit"
    echo "running bf16 path"
elif [[ $PRECISION == "bf32" ]]; then
    ARGS="$ARGS --bf32 --jit"
    echo "running bf32 path"
elif [[ $PRECISION == "fp16" ]]; then
    ARGS="$ARGS --fp16 --jit"
    echo "running fp16 path"
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    ARGS="$ARGS --jit"
    echo "running fp32 path"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf16, bf32, int8, and avx-int8"
    exit 1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
CORES_PER_NUMA=`expr $CORES \* $SOCKETS / $NUMAS`
CORES_PER_INSTANCE=4

if [[ "0" == ${TORCH_INDUCTOR} ]];then
    ARGS_IPEX="$ARGS_IPEX --memory-allocator jemalloc --log_dir="${OUTPUT_DIR}" --log_file_prefix="./${LOG_PREFIX}_${PRECISION}""
else
    ARGS_IPEX="$ARGS_IPEX --disable-numactl --enable-jemalloc --log_path="${OUTPUT_DIR}" "
fi

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    ARGS="$ARGS -e -a resnet50 ../ --dummy"
    ARGS_IPEX="$ARGS_IPEX --throughput_mode"
    BATCH_SIZE=${BATCH_SIZE:-112}
fi

if [[ "$TEST_MODE" == "REALTIME" ]]; then
    NUMBER_INSTANCE=`expr $CORES_PER_NUMA / $CORES_PER_INSTANCE`
    ARGS="$ARGS -e -a resnet50 ../ --dummy --weight-sharing --number-instance $NUMBER_INSTANCE"
    BATCH_SIZE=${BATCH_SIZE:-1}
fi

if [[ "$TEST_MODE" == "ACCURACY" ]]; then
    python ${MODEL_DIR}/../../common/hub_help.py \
        --url https://download.pytorch.org/models/resnet50-0676ba61.pth
    ARGS="$ARGS --pretrained -e -a resnet50 ${DATASET_DIR}"
    BATCH_SIZE=${BATCH_SIZE:-128}
fi

weight_sharing=false
if [ ${WEIGHT_SHARING} ]; then
  echo "Running RN50 inference with runtime extension enabled."
  weight_sharing=true
fi

TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}
if [ "$weight_sharing" = true ]; then

    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    CORES_PER_INSTANCE=$CORES
    INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
    LAST_INSTANCE=`expr $INSTANCES - 1`
    INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
    OMP_NUM_THREADS=$CORES_PER_INSTANCE

    BATCH_PER_STREAM=2
    CORES_PER_STREAM=1
    STREAM_PER_INSTANCE=`expr $CORES / $CORES_PER_STREAM`
    BATCH_SIZE=`expr $BATCH_PER_STREAM \* $STREAM_PER_INSTANCE`

    for i in $(seq 0 $LAST_INSTANCE); do
        numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
        start_core_i=`expr $i \* $CORES_PER_INSTANCE`
        end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
        LOG_i=${LOG_PREFIX}_${PRECISION}_${i}.log
        echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
        numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python -u \
            ${MODEL_DIR}/../../common/main_runtime_extension.py \
            $ARGS \
            --ipex \
            --seed 2020 \
            -j 0 \
            -b $BATCH_SIZE \
            --number-instance $STREAM_PER_INSTANCE \
            --use-multi-stream-module \
            --instance-number $i 2>&1 | tee $LOG_i &
    done
    wait
elif [[ "0" == ${TORCH_INDUCTOR} ]];then
    python -m intel_extension_for_pytorch.cpu.launch \
	    ${ARGS_IPEX} \
        ${MODEL_DIR}/../../common/main.py \
        $ARGS \
        --ipex \
        --seed 2020 \
        -j 0 \
        -b $BATCH_SIZE
else
    echo "Running RN50 inference with torch.compile inductor backend."
    export TORCHINDUCTOR_FREEZING=1
    python -m torch.backends.xeon.run_cpu \
        ${ARGS_IPEX} \
        ${MODEL_DIR}/../../common/main.py \
        $ARGS \
        --inductor \
        --seed 2020 \
        -j 0 \
        -b $BATCH_SIZE
fi
wait

latency="N/A"
throughput="N/A"
accuracy="N/A"

if [[ "$TEST_MODE" == "REALTIME" ]]; then
    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/${LOG_PREFIX}* |sed -e 's/.*Throughput://;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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
    latency=$(grep 'P99 Latency' ${OUTPUT_DIR}/${LOG_PREFIX}* |sed -e 's/.*P99 Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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
    echo "resnet50;"latency";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
    echo "resnet50;"p99_latency";${PRECISION};${BATCH_SIZE};${latency}" | tee -a ${OUTPUT_DIR}/summary.log
elif [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    throughput=$(grep 'Throughput:'  ${OUTPUT_DIR}/${LOG_PREFIX}_${PRECISION}_* |sed -e 's/.*Throughput://;s/[^0-9.]//g' |awk '
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
    echo "resnet50;"throughput";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    accuracy=$(grep 'Accuracy:' ${OUTPUT_DIR}/${LOG_PREFIX}_${PRECISION}_* |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
    echo "resnet50;"accuracy";${PRECISION};${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
fi

echo "resnet50;"throughput";"accuracy";"p99_latency";${PRECISION};${BATCH_SIZE};${throughput};${latency};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: examples per second
- key: latency
  value: $latency
  unit: seconds per example
- key: accuracy
  value: $accuracy
  unit: percentage
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."