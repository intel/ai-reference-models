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
ARGS="$ARGS --use_ipex"
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
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, bf16"
    exit 1
fi

mode="jit"
ARGS="$ARGS --jit_mode"
echo "### running with jit mode"


export OMP_NUM_THREADS=4
CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
BATCH_SIZE=${BATCH_SIZE:-1}
FINETUNED_MODEL=${FINETUNED_MODEL:-"distilbert-base-uncased-distilled-squad"}
if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi
EVAL_SCRIPT=${EVAL_SCRIPT:-"./transformers/examples/pytorch/question-answering/run_qa.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}
rm -rf ${OUTPUT_DIR}/latency_log*
python -m intel_extension_for_pytorch.cpu.launch --latency_mode --enable_jemalloc --log_path=${OUTPUT_DIR} --log_file_prefix="./latency_log_${precision}_${mode}" \
  ${EVAL_SCRIPT} $ARGS \
  --model_name_or_path   ${FINETUNED_MODEL} \
  --dataset_name squad \
  --do_eval \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./tmp \
  --per_device_eval_batch_size $BATCH_SIZE \

CORES_PER_INSTANCE=4
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