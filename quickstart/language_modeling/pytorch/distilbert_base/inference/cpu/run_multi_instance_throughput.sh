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
ARGS="$ARGS --use_ipex --benchmark --perf_begin_iter 10 --perf_run_iters 100"
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
elif [[ "$1" == "bf32" ]]
then
    precision="bf32"
    ARGS="$ARGS --bf32 --auto_kernel_selection"
    echo "### running bf32 mode"
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
    echo "Supported precisions are: fp32, bf32, bf16, int8-fp32, int8-bf16"
    exit 1
fi

mode="jit"
ARGS="$ARGS --jit_mode"
echo "### running with jit mode"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi
if [ -z "${SEQUENCE_LENGTH}" ]; then
  echo "The required environment variable SEQUENCE_LENGTH has not been set, please set the seq_length before running"
  exit 1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
BATCH_SIZE=${BATCH_SIZE:-`expr 4 \* $CORES`}
FINETUNED_MODEL=${FINETUNED_MODEL:-"distilbert-base-uncased-finetuned-sst-2-english"}

EVAL_SCRIPT=${EVAL_SCRIPT:-"./transformers/examples/pytorch/text-classification/run_glue.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}

rm -rf ${OUTPUT_DIR}/throughput_log*
python -m intel_extension_for_pytorch.cpu.launch --ninstance 1 --node_id 0  --enable_jemalloc --log_path=${OUTPUT_DIR} --log_file_prefix="./throughput_log_${path}_${precision}_${mode}" \
  ${EVAL_SCRIPT} $ARGS \
  --model_name_or_path   ${FINETUNED_MODEL} \
  --task_name sst2 \
  --do_eval \
  --max_seq_length ${SEQUENCE_LENGTH} \
  --output_dir ./tmp \
  --per_device_eval_batch_size $BATCH_SIZE \

throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/throughput_log* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
echo ""distilbert-base";"throughput";${precision};${BATCH_SIZE};${throughput}" | tee -a ${WORK_SPACE}/summary.log