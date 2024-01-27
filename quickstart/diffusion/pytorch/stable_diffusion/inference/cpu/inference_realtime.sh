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

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR \${DATASET_DIR} does not exist"
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
elif [ "$1" == "int8-bf16" ]; then
    ARGS="$ARGS --precision=int8-bf16"
    echo "### running int8-bf16 datatype"
elif [ "$1" == "int8-fp32" ]; then
    ARGS="$ARGS --precision=int8-fp32"
    echo "### running int8-fp32 datatype"
elif [ "$1" == "bf32" ]; then
    ARGS="$ARGS --precision=bf32"
    echo "### running bf32 datatype"
elif [ "$1" == "fp32" ]; then
    echo "### running fp32 datatype"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, bf32, fp16, bf16, int8-bf16, int8-fp32"
    exit 1
fi

TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}
if [ "$2" == "eager" ]; then
    echo "### running eager mode"
elif [ "$2" == "ipex-jit" ]; then
    ARGS="$ARGS --ipex --jit"
    echo "### running IPEX JIT mode"
elif [ "$2" == "compile-ipex" ]; then
    ARGS="$ARGS --compile_ipex"
    echo "### running torch.compile with ipex backend"
elif [[ "$2" == "compile-inductor" || "1" == ${TORCH_INDUCTOR} ]]; then
    export TORCHINDUCTOR_FREEZING=1
    ARGS="$ARGS --compile_inductor"
    echo "### running torch.compile with inductor backend"
else
    echo "The specified mode '$2' is unsupported."
    echo "Supported mode are: eager, ipex-jit, compile-ipex, compile-inductor"
    exit 1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
CORES_PER_NUMA=`expr $CORES \* $SOCKETS / $NUMAS`

CORES_PER_INSTANCE=4

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=$CORES_PER_INSTANCE

NUMBER_INSTANCE=`expr $CORES_PER_NUMA / $CORES_PER_INSTANCE`

PRECISION=$1

rm -rf ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_inference_realtime*

python -m intel_extension_for_pytorch.cpu.launch \
    --memory-allocator jemalloc \
    --ninstances $NUMAS \
    --log-dir ${OUTPUT_DIR} \
    --log_file_prefix stable_diffusion_${PRECISION}_inference_realtime \
    ${MODEL_DIR}/models/diffusion/pytorch/stable_diffusion/inference.py \
    --dataset_path=${DATASET_DIR} \
    --benchmark \
    -w 1 -i 1 \
    --weight-sharing \
    --number-instance $NUMBER_INSTANCE \
    $ARGS

# For the summary of results
wait

TOTAL_CORES=`expr $CORES \* $SOCKETS`
INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`

throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_inference_realtime* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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
        printf("%.3f", sum);
}')
echo ""stable_diffusion";"latency";$1;${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
