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

if [[ $TEST_MODE == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
    BATCH_SIZE=${BATCH_SIZE:-448}
    LOG_PREFIX=/rnnt_${PRECISION}_inference_throughput
elif [[ $TEST_MODE == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    BATCH_SIZE=${BATCH_SIZE:-64}
    LOG_PREFIX=/rnnt_${PRECISION}_inference_accuracy
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    echo "TEST_MODE set to REALTIME"
    BATCH_SIZE=${BATCH_SIZE:-1}
    LOG_PREFIX=/rnnt_${PRECISION}_inference_realtime
else
    echo "Please set TEST_MODE to THROUGHPUT or REALTIME or ACCURACY"
    exit
fi

MODEL_DIR=${MODEL_DIR-$PWD}

if [ ! -e "${MODEL_DIR}/inference.py" ]; then
  echo "Could not find the script of inference.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the inference.py exist at the: \${MODEL_DIR}/inference.py"
  exit 1
fi

if [ ! -e "${CHECKPOINT_DIR}/results/rnnt.pt" ]; then
  echo "The pretrained model \${CHECKPOINT_DIR}/results/rnnt.pt does not exist"
  exit 1
fi

if [ ! -d "${DATASET_DIR}/dataset/LibriSpeech" ]; then
  echo "The DATASET_DIR \${DATASET_DIR}/dataset/LibriSpeech does not exist"
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
  echo "Please set PRECISION to fp32, avx-fp32, bf32 or bf16."
  exit 1
fi

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [ "$PRECISION" == "bf16" ]; then
    ARGS="$ARGS --mix-precision"
    echo "### running bf16 datatype"
elif [ "$PRECISION" == "bf32" ]; then
    ARGS="$ARGS --bf32"
    echo "### running bf32 datatype"
else
    echo "### running fp32 datatype"
fi

if [[ $TEST_MODE == "THROUGHPUT" ]]; then
    LOG_PREFIX=/rnnt_${PRECISION}_inference_throughput
    ARGS_IPEX="$ARGS_IPEX --throughput_mode"
    ARGS="$ARGS --warm_up 3 --sort_by_duration"
elif [[ $TEST_MODE == "ACCURACY" ]]; then
    LOG_PREFIX=/rnnt_${PRECISION}_inference_accuracy
    ARGS_IPEX="$ARGS_IPEX --latency_mode"
    ARGS="$ARGS --warm_up 10"
else
    LOG_PREFIX=/rnnt_${PRECISION}_inference_realtime

fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

rm -rf ${OUTPUT}/${LOG_PREFIX}

python -m intel_extension_for_pytorch.cpu.launch \
    --memory-allocator jemalloc \
    ${ARGS_IPEX} \
    --log-dir ${OUTPUT_DIR} \
    --log_file_prefix ${LOG_PREFIX} \
    ${MODEL_DIR}/inference.py \
    --dataset_dir ${DATASET_DIR}/dataset/LibriSpeech/ \
    --val_manifest ${DATASET_DIR}/dataset/LibriSpeech/librispeech-dev-clean-wav.json \
    --model_toml ${MODEL_DIR}/rnnt.toml \
    --ckpt ${CHECKPOINT_DIR}/results/rnnt.pt \
    --batch_size $BATCH_SIZE \
    --ipex \
    --jit \
    $ARGS 2>&1 | tee ${OUTPUT_DIR}/${LOG_PREFIX}.log

wait

latency="N/A"
throughput="N/A"
accuracy="N/A"

if [[ "$TEST_MODE" == "REALTIME" ]]; then
    CORES=`lscpu | grep Core | awk '{print $4}'`
    CORES_PER_INSTANCE=4

    INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET=`expr $CORES / $CORES_PER_INSTANCE`

    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/${LOG_PREFIX}* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET '
    BEGIN {
            sum = 0;
    i = 0;
        }
        {
            sum = sum + $1;
    i++;
        }
    END   {
    sum = sum / i * INSTANCES_PER_SOCKET;
            printf("%.2f", sum);
    }')
    p99_latency=$(grep 'P99 Latency' ${OUTPUT_DIR}/${LOG_PREFIX}* |sed -e 's/.*P99 Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET '
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
    echo ""RNN-T";"latency";$PRECISION; ${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
    echo ""RNN-T";"p99_latency";$PRECISION; ${BATCH_SIZE};${p99_latency}" | tee -a ${OUTPUT_DIR}/summary.log
elif [[ $TEST_MODE == "THROUGHPUT" ]]; then
    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/${LOG_PREFIX}* |sed -e 's/.*Throughput://;s/[^0-9.]//g' |awk '
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
    echo ""RNN-T";"throughput";$PRECISION; ${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
elif [[ $TEST_MODE == "ACCURACY" ]]; then
    accuracy=$(grep 'Accuracy:' ${OUTPUT_DIR}/${LOG_PREFIX}* |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
    WER=$(grep 'Evaluation WER:' ${OUTPUT_DIR}/${LOG_PREFIX}*|sed -e 's/.*Evaluation WER//;s/[^0-9.]//g')
    echo ""RNN-T";"accuracy";$PRECISION; ${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
    echo ""RNN-T";"WER";$PRECISION; ${BATCH_SIZE};${WER}" | tee -a ${work_space}/summary.log
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
