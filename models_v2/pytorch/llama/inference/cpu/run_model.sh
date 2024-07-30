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

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    echo "Running Multi-instance Throughput Inference"
    BATCH_SIZE=${BATCH_SIZE:-1}
    export KMP_BLOCKTIME=1
    rm -rf ${OUTPUT_DIR}/throughput_log*
    export usecase=throughput
    NUM_WARMUP=${NUM_WARMUP:-10}
    NUM_ITER=${NUM_ITER:-20}
    ARGS="$ARGS  --benchmark --num-warmup ${NUM_WARMUP} --num-iter $NUM_ITER --token-latency"
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`

elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    echo "Running Multi-instance Realtime Inference"
    BATCH_SIZE=${BATCH_SIZE:-1}
    export OMP_NUM_THREADS=${CORE_PER_INSTANCE}
    export KMP_BLOCKTIME=1
    rm -rf ${OUTPUT_DIR}/latency_log*
    export usecase=latency
    NUM_WARMUP=${NUM_WARMUP:-10}
    NUM_ITER=${NUM_ITER:-20}
    ARGS="$ARGS  --benchmark --num-warmup ${NUM_WARMUP} --num-iter $NUM_ITER --token-latency"
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`

elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    ARGS="$ARGS --accuracy_only  --lambada"
    rm -rf ${OUTPUT_DIR}/*accuracy*
else
    echo "Please set TEST_MODE to THROUGHPUT, REALTIME or ACCURACY"
    exit
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi

rm -rf ${OUTPUT_DIR}/results.yaml
rm -rf ${OUTPUT_DIR}/summary.log

if [[ "${PRECISION}" == "int8-fp32" ]] || [[ "${PRECISION}" == "int8-fp16"  ]]; then
    if [ ! -f "${OUTPUT_DIR}/qconfig.json" ]; then
    echo "Performing quantization"
    ./do_quantization.sh calibration sq
    fi
fi


echo "### running with intel extension for pytorch"

if [[ "${PRECISION}" == "fp32" ]]
then
    precision="fp32"
    ARGS="$ARGS --dtype fp32 "
    echo "### running fp32 mode"
elif [[ "${PRECISION}" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --dtype bf16 "
    echo "### running bf16 mode"
elif [[ "${PRECISION}" == "fp16" ]]
then
    precision=fp16
    ARGS="$ARGS --dtype fp16"
    echo "### running fp16 mode"
elif [[ "${PRECISION}" == "bf32" ]]
then
    precision="bf32"
    ARGS="$ARGS --dtype bf32"
    echo "### running bf32 mode"
elif [[ "${PRECISION}" == "int8-fp32" ]]
then
    precision="int8-fp32"
    ARGS="$ARGS --dtype int8 --int8-qconfig   ${OUTPUT_DIR}/qconfig.json"
    echo "### running int8-fp32 mode"
elif [[ "${PRECISION}" == "int8-bf16" ]]
then
    precision="int8-bf16"
    ARGS="$ARGS --dtype int8 --int8_bf16_mixed --int8-qconfig ${OUTPUT_DIR}/qconfig.json"
    echo "### running int8-bf16 mode"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, bf32, bf16, fp16, int8-fp32, int8-bf16"
    exit 1
fi


FINETUNED_MODEL=${FINETUNED_MODEL:-"meta-llama/Llama-2-7b-hf"}

EVAL_SCRIPT=${EVAL_SCRIPT:-"${PWD}/run_llm.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}
TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}

if [[ "$TEST_MODE" != "ACCURACY" ]]; then
    if [ -z "${OUTPUT_TOKEN}" ]; then
        echo "The required environment variable OUTPUT_TOKEN has not been set, please set before running, e.g. export OUTPUT_TOKEN=32"
        exit 1
    fi

    if [ -z "${INPUT_TOKEN}" ]; then
        echo "The required environment variable INPUT_TOKEN has not been set, please set before running (choice in 32 64 128 512 1024 2016 ), e.g. export INPUT_TOKEN=1024"
        exit 1
    fi
    
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        path="ipex"
        mode="jit"
        ARGS="$ARGS --jit"
        echo "### running with jit mode"
        if [[ "${PRECISION}" == "int8-bf16" || "${PRECISION}" == "int8-fp32" ]];then
            ARGS="$ARGS --ipex_smooth_quant"
    fi
    python -m intel_extension_for_pytorch.cpu.launch --throughput-mode --memory-allocator tcmalloc --log_dir=${OUTPUT_DIR} --log_file_prefix="./${usecase}_log_${precision}_${mode}" \
        ${EVAL_SCRIPT} $ARGS \
        --ipex \
        -m ${FINETUNED_MODEL} \
        --max-new-tokens ${OUTPUT_TOKEN} \
        --input-tokens  ${INPUT_TOKEN} \
        --batch-size $BATCH_SIZE
    else
        echo "### running with torch.compile inductor backend"
        export TORCHINDUCTOR_FREEZING=1
        python -m torch.backends.xeon.run_cpu --throughput-mode --enable_tcmalloc --log_path=${OUTPUT_DIR} \
            ${EVAL_SCRIPT} $ARGS \
            --inductor \
            -m ${FINETUNED_MODEL} \
            --max-new-tokens ${OUTPUT_TOKEN} \
            --input-tokens  ${INPUT_TOKEN} \
            --batch-size $BATCH_SIZE
    fi

    latency=($(grep -i 'inference-latency:' ${OUTPUT_DIR}/${usecase}_log_${PRECISION}* |sed -e 's/.*atency: //;s/[^0-9.]//g;s/\.$//' |awk '
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

    first_latency=($(grep -i 'first-token-latency:' ${OUTPUT_DIR}/${usecase}_log_${PRECISION}*  |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
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
    rest_token_latency=($(grep -i '^rest-token-latency:' ${OUTPUT_DIR}/${usecase}_log_${PRECISION}*  |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
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
    P90_rest_token_latency=($(grep -i 'P90-rest-token-latency:' ${OUTPUT_DIR}/${usecase}_log_${PRECISION}*  |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
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

    echo "--------------------------------Performance Summary per NUMA Node--------------------------------"
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};latency;"total-latency";${PRECISION};${BATCH_SIZE}; ${latency} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};latency;"first-token-latency";${PRECISION};${BATCH_SIZE}; ${first_latency} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};latency;"rest-token-latency";${PRECISION};${BATCH_SIZE}; ${rest_token_latency} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};latency;"P90-rest-token-latency";${PRECISION};${BATCH_SIZE}; ${P90_rest_token_latency} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};latency;"token_per_sec";${PRECISION};${BATCH_SIZE}; ${token_per_sec} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};latency;"first_token_thp";${PRECISION};${BATCH_SIZE}; ${first_token_thp} " |tee -a ${OUTPUT_DIR}/summary.log

    first_token_latency=$( grep "first-token-latency;" ${OUTPUT_DIR}/summary.log | awk '{print $NF}' )
    rest_token_latency=$( grep ";rest-token-latency;" ${OUTPUT_DIR}/summary.log | awk '{print $NF}' )

    ## Single-socket throughput calculation
    first_token_throughput=$( echo "(1/$first_token_latency)*${BATCH_SIZE}" | bc -l )
    rest_token_throughput=$( echo "(1/$rest_token_latency)*${BATCH_SIZE}" | bc -l )
    accuracy="N/A"

else
    first_token_latency="N/A"
    rest_token_latency="N/A"
    first_token_throughput="N/A"
    rest_token_throughput="N/A"
    BATCH_SIZE=${BATCH_SIZE:-1}
    echo "Running Accuracy Inference"
    rm -rf ${OUTPUT_DIR}/*accuracy*
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        path="ipex"
        mode="jit"
        ARGS="$ARGS --jit"
        echo "### running with jit mode"
        if [[ "${PRECISION}" == "int8-bf16" || "${PRECISION}" == "int8-fp32" ]];then
            ARGS="$ARGS --ipex_smooth_quant"
        fi
        python -m intel_extension_for_pytorch.cpu.launch --nodes-list 0 --memory-allocator tcmalloc --log_dir=${OUTPUT_DIR} --log_file_prefix="./LLaMa_${PRECISION}_accuracy_${mode}" \
            ${EVAL_SCRIPT} $ARGS \
            --ipex \
            --model-name-or-path ${FINETUNED_MODEL}
    else
        echo "### running with torch.compile inductor backend"
        export TORCHINDUCTOR_FREEZING=1
        python -m torch.backends.xeon.run_cpu --node_id 0 --enable_tcmalloc --log_path=${OUTPUT_DIR} \
            ${EVAL_SCRIPT} $ARGS \
            --inductor \
            --model-name-or-path ${FINETUNED_MODEL}
    fi

    accuracy=$(cat ${OUTPUT_DIR}/LLaMa_${PRECISION}_accuracy* | grep "Accuracy:" |sed -e 's/.*= //;s/[^0-9.]//g')

    echo "${FINETUNED_MODEL};"accuracy";${PRECISION};${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
fi

yaml_content=$(cat << EOF
results:
- key: first token throughput
  value: $first_token_throughput
- key: rest token throughput
  value: $rest_token_throughput
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
