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
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

path="ipex"
ARGS="$ARGS --use_ipex --benchmark --perf_begin_iter 500 --perf_run_iters 2000 "
echo "### running with intel extension for pytorch"

precision="fp32"
if [[ "$1" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --mix_bf16"
    echo "### running bf16 mode"
elif [[ "$1" == "fp32" ]]
then
    echo "### running fp32 mode"
elif [[ "$1" == "int8-fp32" ]]
then
    precision="int8-fp32"
    ARGS="$ARGS --int8 --int8_config configure.json"
    echo "### running int8-fp32 mode"
elif [[ "$1" == "int8-bf16" ]]
then
    precision="int8-bf16"
    ARGS="$ARGS --mix_bf16 --int8 --int8_config configure.json"
    echo "### running int8-bf16 mode"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, bf16, int8-fp32, int8-bf16"
    exit 1
fi

if [ -z "${SEQUENCE_LENGTH}" ]; then
  echo "The required environment variable SEQUENCE_LENGTH has not been set, please set the seq_length before running, e.g. export SEQUENCE_LENGTH=128"
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
ARGS="$ARGS --jit_mode"
echo "### running with jit mode"


export OMP_NUM_THREADS=${CORE_PER_INSTANCE}
CORES=`lscpu | grep Core | awk '{print $4}'`
ARGS="$ARGS --use_share_weight --total_cores ${CORES} --cores_per_instance ${OMP_NUM_THREADS}"
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
BATCH_SIZE=${BATCH_SIZE:-1}
FINETUNED_MODEL=${FINETUNED_MODEL:-"distilbert-base-uncased-finetuned-sst-2-english"}

EVAL_SCRIPT=${EVAL_SCRIPT:-"./transformers/examples/pytorch/text-classification/run_glue.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}
rm -rf ${OUTPUT_DIR}/latency_log*
python -m intel_extension_for_pytorch.cpu.launch --ninstance 1 --node_id 0  --enable_jemalloc --log_path=${OUTPUT_DIR} --log_file_prefix="./latency_log_${precision}_${mode}" \
  ${EVAL_SCRIPT} $ARGS \
  --model_name_or_path   ${FINETUNED_MODEL} \
  --task_name sst2 \
  --do_eval \
  --max_seq_length ${SEQUENCE_LENGTH} \
  --output_dir ./tmp \
  --per_device_eval_batch_size $BATCH_SIZE \

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

echo $INSTANCES_PER_SOCKET
echo ""distilbert-base";"latency";${precision};${BATCH_SIZE};${throughput}" | tee -a ${WORK_SPACE}/summary.log