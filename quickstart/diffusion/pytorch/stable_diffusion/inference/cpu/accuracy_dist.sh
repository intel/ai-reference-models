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

if [ "$2" == "eager" ]; then
    echo "### running eager mode"
elif [ "$2" == "ipex-jit" ]; then
    ARGS="$ARGS --ipex --jit"
    echo "### running IPEX JIT mode"
elif [ "$2" == "compile-ipex" ]; then
    ARGS="$ARGS --compile_ipex"
    echo "### running torch.compile with ipex backend"
elif [ "$2" == "compile-inductor" ]; then
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
TOTAL_CORES=`expr $CORES \* $SOCKETS`
NNODES=${NNODES:-1}
HOSTFILE=${HOSTFILE:-./hostfile}
NUM_RANKS=$(( NNODES * SOCKETS ))

if [ ${LOCAL_BATCH_SIZE} ]; then
    GLOBAL_BATCH_SIZE=$(( LOCAL_BATCH_SIZE * NNODES * SOCKETS ))
fi

CORES_PER_INSTANCE=$CORES

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

PRECISION=$1

<< EOF
#oneCCL settings
export CCL_WORKER_COUNT=8
export CCL_LOG_LEVEL=info
export CCL_BF16=avx512bf
export CCL_ATL_TRANSPORT=ofi
export CCL_MNIC_COUNT=2
export CCL_MNIC=local
export CCL_MNIC_NAME=irdma1,irdma5
export CCL_ALLREDUCE=ring
export CCL_WORKER_COUNT=8

for (( i = $SOCKETS; i < 2*$SOCKETS; i++ )); do  # pin CCL workers to HT
  START_CORE=$(( i * CORES ))
  for (( j = 0; j < $CCL_WORKER_COUNT; j++)); do
   CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY} $((START_CORE + j))"
  done
done

export CCL_WORKER_AFFINITY=`echo ${CCL_WORKER_AFFINITY} | tr " " ","`


#DDP settings
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export MASTER_ADDR=`head -1 hostfile`

# Fabric settings
export FI_PROVIDER=psm3
export PSM3_IDENTIFY=1
export PSM3_ALLOW_ROUTERS=1
export PSM3_RDMA=1
export PSM3_PRINT_STATS=0
export PSM3_RV_MR_CACHE_SIZE=8192
export PSM3_KASSIST_MODE=none
#export PSM3_NIC='irdma*
export FI_PSM3_CONN_TIMEOUT=100
EOF

rm -rf ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_dist_inference_accuracy*

oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh

export FI_PROVIDER_PATH=$oneccl_bindings_for_pytorch_path/lib/prov

python -m intel_extension_for_pytorch.cpu.launch \
    --memory-allocator tcmalloc \
    --distributed \
    --nnodes ${NNODES} \
    --hostfile ${HOSTFILE} \
    --logical-cores-for-ccl --ccl_worker_count 8 \
    ${MODEL_DIR}/models/diffusion/pytorch/stable_diffusion/inference.py \
    --dataset_path=${DATASET_DIR} \
    --dist-backend ccl \
    --accuracy \
    $ARGS 2>&1 | tee ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_dist_inference_accuracy.log

# For the summary of results
wait

accuracy=$(grep 'FID:' ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_dist_inference_accuracy* |sed -e 's/.*FID//;s/[^0-9.]//g')
echo ""stable_diffusion";"FID";$1;${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
