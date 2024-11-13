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
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"


echo "### running with intel extension for pytorch"

precision="fp32"
if [[ "$PRECISION" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --bf16"
    echo "### running bf16 mode"
elif [[ "$PRECISION" == "fp16" ]]
then
    precision=fp16
    ARGS="$ARGS --fp16_cpu"
    echo "### running fp16 mode"
elif [[ "$PRECISION" == "fp32" ]]
then
    echo "### running fp32 mode"
elif [[ "$PRECISION" == "bf32" ]]
then
    precision="bf32"
    ARGS="$ARGS --bf32 --auto_kernel_selection"
    echo "### running bf32 mode"
elif [[ "$PRECISION" == "int8-fp32" ]]
then
    precision="int8-fp32"
    ARGS="$ARGS --int8 --int8_config configure.json"
    echo "### running int8-fp32 mode"
elif [[ "$PRECISION" == "int8-bf16" ]]
then
    precision="int8-bf16"
    ARGS="$ARGS --bf16 --int8 --int8_config configure.json"
    echo "### running int8-bf16 mode"
else
    echo "The specified precision '$PRECISION' is unsupported."
    echo "Supported precisions are: fp32, bf32, bf16, int8-fp32, int8-bf16"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi
mkdir -p ${OUTPUT_DIR}

if [ -z "${SEQUENCE_LENGTH}" ]; then
  echo "The required environment variable SEQUENCE_LENGTH has not been set, please set the seq_length before running"
  exit 1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
FINETUNED_MODEL=${FINETUNED_MODEL:-"distilbert-base-uncased-finetuned-sst-2-english"}

EVAL_SCRIPT=${EVAL_SCRIPT:-"./transformers/examples/pytorch/text-classification/run_glue.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    ARGS="$ARGS --benchmark --perf_begin_iter 10 --perf_run_iters 100"
    BATCH_SIZE=${BATCH_SIZE:-`expr 4 \* $CORES`}
    echo "Running throughput"
    rm -rf ${OUTPUT_DIR}/distilbert_throughput*
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    ARGS="$ARGS --benchmark --perf_begin_iter 500 --perf_run_iters 2000"
    export OMP_NUM_THREADS=${CORE_PER_INSTANCE}
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
    CORES_PER_NUMA=`expr $CORES \* $SOCKETS / $NUMAS`
    BATCH_SIZE=${BATCH_SIZE:-1}
    ARGS="$ARGS --use_share_weight --total_cores ${CORES_PER_NUMA} --cores_per_instance ${OMP_NUM_THREADS}"
    echo "Running realtime inference"
    rm -rf ${OUTPUT_DIR}/distilbert_latency*
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    BATCH_SIZE=${BATCH_SIZE:-1}
    echo "Running accuracy"
    rm -rf ${OUTPUT_DIR}/distilbert_accuracy*
fi

TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        path="ipex"
        mode="jit"
        ARGS="$ARGS --jit_mode_eval"
        echo "### running with jit mode"
        python -m intel_extension_for_pytorch.cpu.launch --throughput_mode  --memory-allocator jemalloc --log_dir=${OUTPUT_DIR} --log_file_prefix="./distilbert_throughput_${path}_${precision}_${mode}" \
            ${EVAL_SCRIPT} $ARGS \
            --use_ipex \
            --model_name_or_path   ${FINETUNED_MODEL} \
            --task_name sst2 \
            --do_eval \
            --max_seq_length ${SEQUENCE_LENGTH} \
            --output_dir ${OUTPUT_DIR} \
            --per_device_eval_batch_size $BATCH_SIZE \
            --dataloader_drop_last
    else
        echo "Running inference with torch.compile inductor backend."
        export TORCHINDUCTOR_FREEZING=1
        ARGS="$ARGS --inductor"
        python -m torch.backends.xeon.run_cpu --disable-numactl --throughput-mode  --enable-jemalloc --log-path=${OUTPUT_DIR} \
            ${EVAL_SCRIPT} $ARGS \
            --model_name_or_path   ${FINETUNED_MODEL} \
            --task_name sst2 \
            --do_eval \
            --max_seq_length ${SEQUENCE_LENGTH} \
            --output_dir ${OUTPUT_DIR} \
            --per_device_eval_batch_size $BATCH_SIZE \
            --dataloader_drop_last
    fi

elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        path="ipex"
        echo "### running with intel extension for pytorch"
        mode="jit"
        ARGS="$ARGS --jit_mode_eval"
        echo "### running with jit mode"
        python -m intel_extension_for_pytorch.cpu.launch --log_dir=${OUTPUT_DIR} --log_file_prefix="./distilbert_accuracy_${precision}_${mode}" \
            ${EVAL_SCRIPT} $ARGS \
            --use_ipex \
            --model_name_or_path   ${FINETUNED_MODEL} \
            --task_name sst2 \
            --do_eval \
            --max_seq_length ${SEQUENCE_LENGTH} \
            --output_dir ${OUTPUT_DIR} \
            --per_device_eval_batch_size $BATCH_SIZE
    else
        echo "Running inference with torch.compile inductor backend."
        export TORCHINDUCTOR_FREEZING=1
        ARGS="$ARGS --inductor"
        python -m torch.backends.xeon.run_cpu --disable-numactl --log-path=${OUTPUT_DIR} \
            ${EVAL_SCRIPT} $ARGS \
            --model_name_or_path   ${FINETUNED_MODEL} \
            --task_name sst2 \
            --do_eval \
            --max_seq_length ${SEQUENCE_LENGTH} \
            --output_dir ${OUTPUT_DIR} \
            --per_device_eval_batch_size $BATCH_SIZE
    fi
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        path="ipex"
        mode="jit"
        ARGS="$ARGS --jit_mode_eval"
        echo "### running with jit mode"
        python -m intel_extension_for_pytorch.cpu.launch --ninstances $NUMAS --memory-allocator jemalloc --log_dir=${OUTPUT_DIR} --log_file_prefix="./distilbert_latency_${precision}_${mode}" \
             ${EVAL_SCRIPT} $ARGS \
             --use_ipex \
             --model_name_or_path   ${FINETUNED_MODEL} \
             --task_name sst2 \
             --do_eval \
             --max_seq_length ${SEQUENCE_LENGTH} \
             --output_dir ${OUTPUT_DIR} \
             --per_device_eval_batch_size $BATCH_SIZE
    else
        echo "Running inference with torch.compile inductor backend."
        export TORCHINDUCTOR_FREEZING=1
        ARGS="$ARGS --inductor"
        python -m torch.backends.xeon.run_cpu --disable-numactl --ninstances $NUMAS --enable-jemalloc --log-path=${OUTPUT_DIR} \
             ${EVAL_SCRIPT} $ARGS \
             --model_name_or_path   ${FINETUNED_MODEL} \
             --task_name sst2 \
             --do_eval \
             --max_seq_length ${SEQUENCE_LENGTH} \
             --output_dir ${OUTPUT_DIR} \
             --per_device_eval_batch_size $BATCH_SIZE
    fi
fi

if [[ "$TEST_MODE" == "REALTIME" ]]; then
    CORES_PER_INSTANCE=${OMP_NUM_THREADS}
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
    INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
fi

throughput="N/A"
latency="N/A"
accuracy="N/A"
if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    # Capture and aggregate throughput values
    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/distilbert_throughput* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
    latency=$(grep 'P99 Latency' ${OUTPUT_DIR}/distilbert_throughput* |sed -e 's/.*P99 Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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
    echo "--------------------------------Performance Summary per NUMA Node--------------------------------"
    echo ""distilbert-base";"throughput";${precision};${BATCH_SIZE};${throughput}" | tee -a ${WORK_SPACE}/summary.log
    echo ""distilbert-base";"p99_latency";${precision};${BATCH_SIZE};${latency}" | tee -a ${WORK_SPACE}/summary.log
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    # Capture and aggregate latency values
    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/distilbert_throughput* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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

    latency=$(grep 'P99 Latency' ${OUTPUT_DIR}/distilbert_latency* |sed -e 's/.*P99 Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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
    echo $INSTANCES_PER_SOCKET
    echo ""distilbert-base";"latency";${precision};${BATCH_SIZE};${throughput}" | tee -a ${WORK_SPACE}/summary.log
    echo ""distilbert-base";"p99_latency";${precision};${BATCH_SIZE};${latency}" | tee -a ${WORK_SPACE}/summary.log
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    # Capture and aggregate throughput values
    throughput=$(grep 'eval_samples_per_second' ${OUTPUT_DIR}/distilbert_accuracy* | sed -e 's/.*eval_samples_per_second\s*=\s*//;s/[^0-9.]//g' | awk '{
        sum += $1;
        count++;
    } END {
        if (count > 0) {
            avg = sum / count;
            printf "%.2f", avg;
        }
    }')

    # Calculate latency based on throughput
    if [ -n "$throughput" ]; then
        latency=$(echo "1 / $throughput * 1000" | bc -l)
        latency=$(printf "%.5f" $latency)
    else
        latency="0"  # Handle the case where throughput is not available
    fi

    # Capture and aggregate accuracy values
    accuracy=$(cat ${OUTPUT_DIR}/accuracy_log* | grep "eval_accuracy" |sed -e 's/.*= //;s/[^0-9.]//g')
    f1=$(cat ${OUTPUT_DIR}/accuracy_log* | grep "eval_f1" |sed -e 's/.*= //;s/[^0-9.]//g')
    echo ""distilbert-base";"accuracy";${precision};${BATCH_SIZE};${accuracy}" | tee -a ${WORK_SPACE}/summary.log

fi

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: sentences per second
- key: latency
  value: $latency
  unit: ms
- key: accuracy
  value: $accuracy
  unit: percentage
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
