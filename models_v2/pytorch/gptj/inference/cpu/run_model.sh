#!/bin/bash

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
LOG_PREFIX=""

MODEL_DIR=${MODEL_DIR-$PWD}

if [[ "${TEST_MODE}" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
    NUM_ITER=${NUM_ITER:-20}
    LOG_PREFIX="throughput_log"
    ARGS="$ARGS  --benchmark --num-warmup 10 --num-iter $NUM_ITER --token-latency"
elif [[ "${TEST_MODE}" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    LOG_PREFIX="accuracy_log"
elif [[ "${TEST_MODE}" == "REALTIME" ]]; then
    echo "TEST_MODE set to REALTIME"
    LOG_PREFIX="realtime_log"
    NUM_ITER=${NUM_ITER:-20}
    ARGS="$ARGS  --benchmark --num-warmup 10 --num-iter $NUM_ITER --token-latency"
else
    echo "Please set TEST_MODE to THROUGHPUT or REALTIME or ACCURACY"
    exit
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi

if [[ "${PRECISION}" == "fp32" ]]; then
    ARGS="$ARGS --dtype 'fp32' "
    echo "### running fp32 mode"
elif [[ "${PRECISION}" == "bf16" ]]; then
    ARGS="$ARGS --dtype 'bf16' "
    echo "### running bf16 mode"
elif [[ "${PRECISION}" == "fp32" ]]; then
    echo "### running fp32 mode"
elif [[ "${PRECISION}" == "fp16" ]]; then
    ARGS="$ARGS --dtype 'fp16'"
    echo "### running fp16 mode"
elif [[ "${PRECISION}" == "bf32" ]]; then
    ARGS="$ARGS --dtype 'bf32'"
    echo "### running bf32 mode"
elif [[ "${PRECISION}" == "int8-fp32" ]]; then
    ARGS="$ARGS --dtype 'int8' --int8-qconfig  '${MODEL_DIR}/qconfig.json'"
    echo "### running int8-fp32 mode"
elif [[ "${PRECISION}" == "int8-bf16" ]]; then
    ARGS="$ARGS --dtype 'int8' --int8_bf16_mixed --int8-qconfig '${MODEL_DIR}/qconfig.json'"
    echo "### running int8-bf16 mode"
elif [[ "${PRECISION}" == "fp8" ]]; then
    if [[ "${TEST_MODE}" == "ACCURACY" ]]; then
        ARGS="$ARGS --dtype 'fp8' --fp8-config '${MODEL_DIR}/fp8_state_dict.pt'"
        echo "### running fp8 mode"
    else
        echo "fp8 is only supported for ACCURACY in TEST_MODE, please change TEST_MODE to run this precision"
        exit 1
    fi
else
    echo "The specified precision '${PRECISION}' is unsupported."
    if [[ "${TEST_MODE}" == "ACCURACY" ]]; then
        echo "Supported precisions are: fp32, bf32, bf16, fp16, int8-fp32, int8-bf16, and fp8"
    else
        echo "Supported precisions are: fp32, bf32, bf16, fp16, int8-fp32, int8-bf16"
    fi
    exit 1
fi

if [[ "${TEST_MODE}" != "ACCURACY" ]]; then
    if [ -z "${OUTPUT_TOKEN}" ]; then
        echo "The required environment variable OUTPUT_TOKEN has not been set, please set before running, e.g. export OUTPUT_TOKEN=32"
        exit 1
    fi
    if [ -z "${INPUT_TOKEN}" ]; then
        echo "The required environment variable INPUT_TOKEN has not been set, please set before running (choice in 32 64 128 512 1024 2016 ), e.g. export INPUT_TOKEN=1024"
        exit 1
    fi

    export OMP_NUM_THREADS=${CORES_PER_INSTANCE}
    export KMP_BLOCKTIME=-1
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    BATCH_SIZE=${BATCH_SIZE:-1}

    ARGS_IPEX="${ARGS_IPEX} --throughput-mode"
    ARGS="${ARGS} --ipex --max-new-tokens ${OUTPUT_TOKEN} --input-tokens  ${INPUT_TOKEN} --batch-size $BATCH_SIZE"
else
    ARGS_IPEX="${ARGS_IPEX} --nodes-list 0"
fi

EVAL_SCRIPT=${EVAL_SCRIPT:-"${MODEL_DIR}/run_llm.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}
FINETUNED_MODEL=${FINETUNED_MODEL:-"'EleutherAI/gpt-j-6b'"}

TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}

if [[ "0" == ${TORCH_INDUCTOR} ]];then
    path="ipex"
    MODE="jit"
    ARGS="$ARGS --jit --ipex"
    echo "### running with jit mode"
    if [[ "$1" == "int8-bf16" || "$1" == "int8-fp32" ]];then
        ARGS="$ARGS --ipex_smooth_quant"
    fi
    python -m intel_extension_for_pytorch.cpu.launch ${ARGS_IPEX} --memory-allocator tcmalloc --log_dir=${OUTPUT_DIR} --log_file_prefix="./latency_log_${precision}_${mode}" \
        ${EVAL_SCRIPT} $ARGS \
        --model-name-or-path ${FINETUNED_MODEL}
else
    export TORCHINDUCTOR_FREEZING=1
    echo "### running with torch.compile inductor backend"
    python -m intel_extension_for_pytorch.cpu.launch ${ARGS_IPEX} --memory-allocator tcmalloc --log_dir=${OUTPUT_DIR} --log_file_prefix="./latency_log_${precision}_${mode}" \
        ${EVAL_SCRIPT} $ARGS \
        --inductor \
        --model-name-or-path ${FINETUNED_MODEL}
fi

wait

if [[ "${TEST_MODE}" == "ACCURACY" ]]; then
    accuracy=$(cat ${OUTPUT_DIR}/${LOG_PREFIX}_${PRECISION}* | grep "Accuracy:" |sed -e 's/.*= //;s/[^0-9.]//g')
    latency=($(grep -i 'Latency' ${OUTPUT_DIR}/${LOG_PREFIX}_${PRECISION}* |sed -e 's/.*Latency (sec): //;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0  0");
            }
        }
    '))
else
    latency=($(grep -i 'inference-latency:' ${OUTPUT_DIR}/${LOG_PREFIX}_${PRECISION}* |sed -e 's/.*Latency: //;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0  0");
            }
        }
    '))
    first_latency=($(grep -i 'first-token-latency:' ${OUTPUT_DIR}/${LOG_PREFIX}_${PRECISION}*  |sed -e 's/.*Latency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    rest_token_latency=($(grep -i '^rest-token-latency:' ${OUTPUT_DIR}/${LOG_PREFIX}_${PRECISION}*  |sed -e 's/.*Latency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    P90_rest_token_latency=($(grep -i 'P90-rest-token-latency:' ${OUTPUT_DIR}/${LOG_PREFIX}_${PRECISION}*  |sed -e 's/.*Latency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))

    token_per_sec=($(awk -v output_token=$OUTPUT_TOKEN -v total=$latency -v batch=$BATCH_SIZE -v first_token=${first_latency}} '
        BEGIN {
            thp = batch*(output_token-1)/(total-first_token);
            printf("%.3f", thp);
        }
    '))

    first_token_thp=($(awk -v output_token=$OUTPUT_TOKEN -v total=$latency -v batch=$BATCH_SIZE -v first_token=${first_latency}} '
        BEGIN {
            thp = batch*(1)/(first_token);
            printf("%.3f", thp);
        }
    '))
fi

if [[ -z $throughput ]]; then
    throughput="N/A"
fi
if [[ -z $accuracy ]]; then
    accuracy="N/A"
fi
if [[ -z $latency ]]; then
    latency="N/A"
fi

echo ""gptj";"throughput";"accuracy";"latency";${PRECISION};${throughput};${accuracy};${latency}" | tee -a ${OUTPUT_DIR}/summary.log

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: samples/sec
- key: latency
  value: $latency
  unit: s
- key: accuracy
  value: $accuracy
  unit: FID
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
