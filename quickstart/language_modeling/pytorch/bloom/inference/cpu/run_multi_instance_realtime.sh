#!/bin/bash

#
# Copyright (c) 2021 Intel Corporation
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

path="ipex"
ARGS="$ARGS --use_ipex --benchmark --perf_begin_iter 5 --perf_run_iters 15 "
echo "### running with intel extension for pytorch"

precision="fp32"
if [[ "$1" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --bf16"
    echo "### running bf16 mode"
elif [[ "$1" == "fp32" ]]
then
    echo "### running fp32 mode"
elif [[ "$1" == "fp16" ]]
then
    precision=fp16
    ARGS="$ARGS --fp16"
    echo "### running fp16 mode"
elif [[ "$1" == "bf32" ]]
then
    precision="bf32"
    ARGS="$ARGS --bf32"
    echo "### running bf32 mode"
elif [[ "$1" == "int8-fp32" ]]
then
    precision="int8-fp32"
    ARGS="$ARGS --int8 --int8_config configure.json"
    echo "### running int8-fp32 mode"
elif [[ "$1" == "int8-bf16" ]]
then
    precision="int8-bf16"
    ARGS="$ARGS --bf16 --int8 --int8_config configure.json"
    echo "### running int8-bf16 mode"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, bf32, bf16, int8-fp32, int8-bf16"
    exit 1
fi

if [ -z "${CORE_PER_INSTANCE}" ]; then
  echo "The required environment variable CORE_PER_INSTANCE has not been set, please set the cores_per_instance before running, e.g. export CORE_PER_INSTANCE=4"
  exit 1
fi
if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi

mode="jit"
ARGS="$ARGS --jit_mode_eval"
echo "### running with jit mode"


export OMP_NUM_THREADS=${CORE_PER_INSTANCE}
CORES=`lscpu | grep Core | awk '{print $4}'`
ARGS="$ARGS --use_share_weight --total_cores ${CORES} --cores_per_instance ${OMP_NUM_THREADS}"
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
BATCH_SIZE=${BATCH_SIZE:-1}
rm -rf ${OUTPUT_DIR}/latency_log*
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"bigscience/bloom"}
EVAL_SCRIPT=${EVAL_SCRIPT:-"../../../../../../models/language_modeling/pytorch/bloom/run_clm.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}
rm -rf ${OUTPUT_DIR}/accuracy_log*
python -m intel_extension_for_pytorch.cpu.launch --ninstance 1 --node_id 0 --log_path=${OUTPUT_DIR} --log_file_prefix="latency_log_${precision}_${mode}" \
  ${EVAL_SCRIPT} $ARGS \
  --model_name_or_path ${PRETRAINED_MODEL} \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --do_eval \
  --output_dir ${OUTPUT_DIR} \

CORES_PER_INSTANCE=${OMP_NUM_THREADS}
TOTAL_CORES=`expr $CORES \* $SOCKETS`
INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`

throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/latency_log* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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

p99_latency=$(grep 'P99 Latency' ${OUTPUT_DIR}/latency_log* |sed -e 's/.*P99 Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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

echo $INSTANCES_PER_SOCKET
echo ""gptj-6b";"latency";${precision};${BATCH_SIZE};${throughput}" | tee -a ${WORK_SPACE}/summary.log
echo ""gptj-6b";"p99_latency";${precision};${BATCH_SIZE};${p99_latency}" | tee -a ${WORK_SPACE}/summary.log
