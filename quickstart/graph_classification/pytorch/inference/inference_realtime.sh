#!/usr/bin/env bash
#
# Copyright (c) 2023 Intel Corporation
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

MODEL_DIR=${MODEL_DIR-$PWD}

if [ ! -e "${MODEL_DIR}/models/graph_classification/pytorch/inference/inference.py" ]; then
  echo "Could not find the script of inference.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the inference.py exist at the: \${MODEL_DIR}/models/graph_classification/pytorch/inference/inference.py"
  exit 1
fi

if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "The OUTPUT_DIR '${OUTPUT_DIR}' does not exist"
  exit 1
fi

if [[ "$1" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""
if [ "$1" == "bf16" ]; then
    ARGS="$ARGS --bf16"
    echo "### running bf16 datatype"
elif [ "$1" == "fp16" ]; then
    ARGS="$ARGS --fp16"
    echo "### running fp16 datatype"
else
    echo "### running fp32 datatype"
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

BATCH_SIZE=1
PRECISION=$1

rm -rf ${OUTPUT_DIR}/rgat_${PRECISION}_inference_realtime*

python -m intel_extension_for_pytorch.cpu.launch \
    --use_default_allocator \
    --latency_mode \
    --log_path ${OUTPUT_DIR} \
    --log_file_prefix rgat_${PRECISION}_inference_realtime \
    ${MODEL_DIR}/models/graph_classification/pytorch/inference/inference.py \
    --datasets=ogbn-mag \
    --models=rgat \
    --num-layers=1 \
    --eval-batch-sizes=$BATCH_SIZE \
    --num-hidden-channels=64 \
    --num-steps=100 \
    --ipex \
    $ARGS

# For the summary of results
wait

CORES=`lscpu | grep Core | awk '{print $4}'`
CORES_PER_INSTANCE=4

INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET=`expr $CORES / $CORES_PER_INSTANCE`

throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/rgat_${PRECISION}_inference_realtime* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET '
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
echo ""RNN-T";"latency";$1; ${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
