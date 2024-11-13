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

ARGS=${ARGS:-""}
ARGS_IPEX=${ARGS_IPEX:-""}

MODEL_DIR=${MODEL_DIR-$PWD}
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

if [[ "${TEST_MODE}" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
    LOG_PREFIX="throughput_log"
    NUM_ITER=${NUM_ITER:-20}
    ARGS="$ARGS  --benchmark --num-warmup 10 --num-iter $NUM_ITER --token-latency"
elif [[ "${TEST_MODE}" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    LOG_PREFIX="accuracy_log"
    ARGS="$ARGS --accuracy_only  --lambada"
elif [[ "${TEST_MODE}" == "REALTIME" ]]; then
    echo "TEST_MODE set to REALTIME"
    LOG_PREFIX="realtime_log"
    NUM_ITER=${NUM_ITER:-20}
    ARGS="$ARGS  --benchmark --num-warmup 10 --num-iter $NUM_ITER --token-latency"
    export OMP_NUM_THREADS=${CORE_PER_INSTANCE}
else
    echo "Please set TEST_MODE to THROUGHPUT, ACCURACY, OR REALTIME"
    exit 1
fi

if [[ "$PRECISION" == "int8-fp32" ]] || [[ "$PRECISION" == "int8-fp16"  ]]; then
    if [ ! -f "${OUTPUT_DIR}/qconfig-chatglm.json" ]; then
    echo "Performing quantization"
    ./do_quantization.sh calibration sq
    fi
fi

if [[ "${PRECISION}" == "fp32" ]]
then
    ARGS="$ARGS --dtype 'fp32' "
    echo "### running fp32 mode"
elif [[ "${PRECISION}" == "bf16" ]]
then
    ARGS="$ARGS --dtype 'bf16' "
    echo "### running bf16 mode"
elif [[ "${PRECISION}" == "fp32" ]]
then
    echo "### running fp32 mode"
elif [[ "${PRECISION}" == "fp16" ]]
then
    ARGS="$ARGS --dtype 'fp16'"
    echo "### running fp16 mode"
elif [[ "${PRECISION}" == "bf32" ]]
then
    ARGS="$ARGS --dtype 'bf32'"
    echo "### running bf32 mode"
elif [[ "${PRECISION}" == "int8-fp32" ]]
then
    ARGS="$ARGS --dtype 'int8' --int8-qconfig '${OUTPUT_DIR}/qconfig-chatglm.json'"
    echo "### running int8-fp32 mode"
elif [[ "${PRECISION}" == "int8-bf16" ]]
then
    ARGS="$ARGS --dtype 'int8' --int8_bf16_mixed --int8-qconfig '${OUTPUT_DIR}/qconfig-chatglm.json'"
    echo "### running int8-bf16 mode"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, bf32, bf16, fp16, int8-fp32, int8-bf16"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi

if [[ "${TEST_MODE}" == "THROUGHPUT" || "${TEST_MODE}" == "REALTIME" ]]; then
    if [ -z "${OUTPUT_TOKEN}" ]; then
        echo "The required environment variable OUTPUT_TOKEN has not been set, please set before running, e.g. export OUTPUT_TOKEN=32"
        exit 1
    fi
    if [ -z "${INPUT_TOKEN}" ]; then
        echo "The required environment variable INPUT_TOKEN has not been set, please set before running (choice in 32 64 128 512 1024 2016 ), e.g. export INPUT_TOKEN=1024"
        exit 1
    fi

    export KMP_BLOCKTIME=-1
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    BATCH_SIZE=${BATCH_SIZE:-1}
    ARGS="$ARGS --max-new-tokens ${OUTPUT_TOKEN} --input-tokens ${INPUT_TOKEN} --batch-size ${BATCH_SIZE}"
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        ARGS_IPEX="$ARGS_IPEX --throughput-mode --memory-allocator tcmalloc --log_dir=${OUTPUT_DIR}"
    else
        ARGS_IPEX="$ARGS_IPEX --throughput-mode --enable-tcmalloc --log-path=${OUTPUT_DIR}"
    fi
else
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        ARGS_IPEX="$ARGS_IPEX --log_dir=${OUTPUT_DIR}"
    else
        ARGS_IPEX="$ARGS_IPEX --log-path=${OUTPUT_DIR}"
    fi
fi

EVAL_SCRIPT=${MODEL_DIR}/run_llm.py
FINETUNED_MODEL=${FINETUNED_MODEL:-"THUDM/chatglm3-6b"}
rm -rf ${OUTPUT_DIR}/*${LOG_PREFIX}*
TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}
if [[ "0" == ${TORCH_INDUCTOR} ]];then
    path="ipex"
    mode="jit"
    ARGS="$ARGS --jit"
    ARGS_IPEX="$ARGS_IPEX --log_file_prefix=./ChatGLM_${PRECISION}_${LOG_PREFIX}_${mode}"
    echo "### running with jit mode"
    if [[ "$PRECISION" == "int8-bf16" || "$PRECISION" == "int8-fp32" ]];then
        ARGS="$ARGS --ipex_smooth_quant"
    fi
    python -m intel_extension_for_pytorch.cpu.launch ${ARGS_IPEX} \
        ${EVAL_SCRIPT} $ARGS \
        --ipex \
        --model-name-or-path ${FINETUNED_MODEL}
else
    echo "### running with torch.compile inductor backend"
    export TORCHINDUCTOR_FREEZING=1
    python -m torch.backends.xeon.run_cpu --disable-numactl ${ARGS_IPEX} \
        ${EVAL_SCRIPT} $ARGS \
        --inductor \
        --model-name-or-path ${FINETUNED_MODEL}
fi

wait

first_token_latency="N/A"
rest_token_latency="N/A"

if [[ "${TEST_MODE}" == "ACCURACY" ]]; then
    accuracy=$(cat ${OUTPUT_DIR}/ChatGLM_${PRECISION}_accuracy* | grep "Accuracy:" |sed -e 's/.*= //;s/[^0-9.]//g')
    echo "${FINETUNED_MODEL};"accuracy";${precision};${BATCH_SIZE};${accuracy}" | tee -a ${WORK_SPACE}/summary.log
else
    latency=($(grep -i 'inference-latency:' ${OUTPUT_DIR}/ChatGLM_${PRECISION}_${LOG_PREFIX}* |sed -e 's/.*inference-latency: //;s/[^0-9.]//g;s/\.$//' |awk '
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
    first_token_latency=($(grep -i 'first-token-latency:' ${OUTPUT_DIR}/ChatGLM_${PRECISION}_${LOG_PREFIX}*  |sed -e 's/.*first-token-latency: //;s/[^0-9.]//g;s/\.$//' |awk '
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
    rest_token_latency=($(grep -i 'rest-token-latency:' ${OUTPUT_DIR}/ChatGLM_${PRECISION}_${LOG_PREFIX}* |sed -e 's/.*rest-token-latency: //;s/[^0-9.]//g;s/\.$//' |awk '
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
    P90_rest_token_latency=($(grep -i 'P90-rest-token-latency:' ${OUTPUT_DIR}/ChatGLM_${PRECISION}_${LOG_PREFIX}* |sed -e 's/.*P90-rest-token-latency: //;s/[^0-9.]//g;s/\.$//' |awk '
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

    token_per_sec=($(awk -v output_token=$OUTPUT_TOKEN -v total=$latency -v batch=$BATCH_SIZE -v first_token=${first_token_latency}} '
        BEGIN {
            thp = batch*(output_token-1)/(total-first_token);
            printf("%.3f", thp);
        }
    '))

    first_token_thp=($(awk -v output_token=$OUTPUT_TOKEN -v total=$latency -v batch=$BATCH_SIZE -v first_token=${first_token_latency}} '
        BEGIN {
            thp = batch*(1)/(first_token);
            printf("%.3f", thp);
        }
    '))
    echo "--------------------------------Performance Summary per NUMA Node--------------------------------"
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};${LOG_PREFIX};"total-latency";${PRECISION};${BATCH_SIZE}; ${latency} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};${LOG_PREFIX};"first-token-latency";${PRECISION};${BATCH_SIZE}; ${first_token_latency} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};${LOG_PREFIX};"rest-token-latency";${PRECISION};${BATCH_SIZE}; ${rest_token_latency} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};${LOG_PREFIX};"P90-rest-token-latency";${PRECISION};${BATCH_SIZE}; ${P90_rest_token_latency} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};${LOG_PREFIX};"token_per_sec";${PRECISION};${BATCH_SIZE}; ${token_per_sec} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};${LOG_PREFIX};"first_token_thp";${PRECISION};${BATCH_SIZE}; ${first_token_thp} " |tee -a ${OUTPUT_DIR}/summary.log
fi


yaml_content=$(cat << EOF
results:
- key: first token latency
  value: $first_token_latency
- key: rest token latency
  value: $rest_token_latency
- key: accuracy
  value: $accuracy
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
