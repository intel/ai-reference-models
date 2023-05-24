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

if [ ! -e "${MODEL_DIR}/models/diffusion/pytorch/stable_diffusion/inference.py" ]; then
  echo "Could not find the script of inference.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the inference.py exist at the: \${MODEL_DIR}/models/diffusion/pytorch/stable_diffusion/inference.py"
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
    ARGS="$ARGS --precision=bf16"
    echo "### running bf16 datatype"
elif [ "$1" == "fp16" ]; then
    ARGS="$ARGS --precision=fp16"
    echo "### running fp16 datatype"
elif [ "$1" == "int8" ]; then
    ARGS="$ARGS --precision=int8"
    echo "### running int8 datatype"
elif [ "$1" == "int8-bf16" ]; then
    ARGS="$ARGS --precision=int8-bf16"
    echo "### running int8-bf16 datatype"
elif [ "$1" == "fp32" ]; then
    echo "### running fp32 datatype"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, fp16, bf16, int8, int8-bf16"
    exit 1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

PRECISION=$1

rm -rf ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_inference_realtime*

python -m intel_extension_for_pytorch.cpu.launch \
    --enable_jemalloc \
    --latency_mode \
    --log_path ${OUTPUT_DIR} \
    --log_file_prefix stable_diffusion_${PRECISION}_inference_realtime \
    ${MODEL_DIR}/models/diffusion/pytorch/stable_diffusion/inference.py \
    --ipex \
    --jit \
    --benchmark \
    $ARGS

# For the summary of results
wait

CORES=`lscpu | grep Core | awk '{print $4}'`
CORES_PER_INSTANCE=4

INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET=`expr $CORES / $CORES_PER_INSTANCE`

throughput=$(grep '50/50' ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_inference_realtime* | sed -e 's/[^0-9. ]*//g' | grep -oE "[^ ]+$" |awk -v INSTANCES_PER_SOCKET=$INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET '
BEGIN {
        sum = 0;
i = 0;
      }
      {
        sum = sum + (1 / $1);
i++;
      }
END   {
sum = sum / i * INSTANCES_PER_SOCKET;
        printf("%.2f", sum);
}')
echo ""stable_diffusion";"latency";$1;${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
