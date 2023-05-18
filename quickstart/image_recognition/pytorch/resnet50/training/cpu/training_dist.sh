#!/usr/bin/env bash
#
# Copyright (c) 2022 Intel Corporation
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

if [ ! -e "${MODEL_DIR}/models/image_recognition/pytorch/common/main.py"  ]; then
    echo "Could not find the script of main.py. Please set environment variable '\${MODEL_DIR}'."
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${TRAINING_EPOCHS}" ]; then
  echo "The required environment variable TRAINING_EPOCHS has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${GLOBAL_BATCH_SIZE}" ] && [ -z "${LOCAL_BATCH_SIZE}" ]; then
  echo "The required environment variable GLOBAL_BATCH_SIZE or LOCAL_BATCH_SIZE has not been set"
  exit 1
fi

if [ ${GLOBAL_BATCH_SIZE} ] && [ ${LOCAL_BATCH_SIZE} ]; then
  echo "For the required environment variables GLOBAL_BATCH_SIZE and LOCAL_BATCH_SIZE , set only one of them"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, avx-fp32, or bf16."
  exit 1
fi

if [ -z "${MASTER_ADDR}" ]; then
  echo "The required environment variable MASTER_ADDR has not been set"
  exit 1
fi

ARGS=""
ARGS="$ARGS -a resnet50 ${DATASET_DIR}"

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    echo "running bf16 path"
elif [[ $PRECISION == "bf32" ]]; then
    ARGS="$ARGS --bf32"
    echo "running bf32 path"
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    echo "running fp32 path"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf16, bf32"
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
export USE_IPEX=1
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0


export CCL_MNIC=global
export CCL_MNIC_NAME=rocep56s0,rocep73s0,rocep152s0,rocep216s0 #rocep56s0,rocep59s0,rocep73s0,rocep76s0,rocep152s0,rocep155s0,rocep216s0,rocep219s0
export CCL_MNIC_COUNT=4
export PSM3_PRINT_STATS=0
export FI_PROVIDER=psm3
export CCL_ALLREDUCE=rabenseifner
export PSM3_IDENTIFY=1
export PSM3_IDENTIFY=1
export PSM3_ALLOW_ROUTERS=1
export PSM3_RDMA=1
export PSM3_RV_MR_CACHE_SIZE=8192
export FI_PROVIDER_PATH=/usr/lib64/libfabric


rm -rf ${OUTPUT_DIR}/resnet50_dist_training_log_*

oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh

python -m intel_extension_for_pytorch.cpu.launch \
    --use_default_allocator \
    --distributed \
    --nnodes ${NNODES} \
    --hostfile ${HOSTFILE} \
    --nproc_per_node ${SOCKETS} \
    --ncore_per_instance ${CORES_PER_INSTANCE} \
    --logical_core_for_ccl --ccl_worker_count 8 \
    ${MODEL_DIR}/models/image_recognition/pytorch/common/train.py \
    $ARGS \
    --epochs $TRAINING_EPOCHS \
    --warmup-epochs 2  \
    --ipex \
    -j 0 \
    -b $GLOBAL_BATCH_SIZE \
    --seed 2020 \
    --dist-backend ccl \
    --base-op=LARS \
    --base-lr 10.5 \
    --weight-decay 0.00005 2>&1 | tee ${OUTPUT_DIR}/resnet50_dist_training_log_${PRECISION}.log
# For the summary of results
wait

throughput=$(grep 'Training throughput:' ${OUTPUT_DIR}/resnet50_dist_training_log_${PRECISION}.log |sed -e 's/.*Training throughput//;s/[^0-9.]//g' |awk '
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

echo "resnet50;"training distributed throughput";${PRECISION};${GLOBAL_BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
