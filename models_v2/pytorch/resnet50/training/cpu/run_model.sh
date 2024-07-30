#!/usr/bin/env bash
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

MODEL_DIR=${MODEL_DIR-$PWD}

if [ ! -e "${MODEL_DIR}/../../common/main.py"  ]; then
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

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, avx-fp32, bf16, bf32, or fp16."
  exit 1
fi

ARGS=""
ARGS="$ARGS -a resnet50 ${DATASET_DIR}"
ARGS_IPEX=""

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
CORES_PER_INSTANCE=$CORES

if [[ "$DISTRIBUTED" == "true" || "$DISTRIBUTED" == "True" || "$DISTRIBUTED" == "TRUE" ]]; then

    if [ -z "${LOCAL_BATCH_SIZE}" ]; then
        echo "The required environment variable LOCAL_BATCH_SIZE has not been set"
        exit 1
    fi

    if [ -z "${MASTER_ADDR}" ]; then
        echo "The required environment variable MASTER_ADDR has not been set"
        exit 1
    fi

    NNODES=${NNODES:-1}
    HOSTFILE=${HOSTFILE:-./hostfile}
    NUM_RANKS=$(( NNODES * SOCKETS ))
    LOG_PREFIX=resnet50_dist_training_log

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
    export PSM3_HAL=sockets

    oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
    source $oneccl_bindings_for_pytorch_path/env/setvars.sh

    ARGS="$ARGS --warmup-epochs 2 -b $LOCAL_BATCH_SIZE --dist-backend ccl --base-op=LARS --base-lr 10.5 --weight-decay 0.00005"
    ARGS_IPEX="$ARGS_IPEX --memory-allocator tcmalloc --distributed --nnodes ${NNODES} --hostfile ${HOSTFILE} -logical_cores_for_ccl --ccl_worker_count 8"
else
    BATCH_SIZE=${BATCH_SIZE:-128}
    ARGS_IPEX="$ARGS_IPEX --ninstances 1  --ncore_per_instance ${CORES_PER_INSTANCE} --log_path=${OUTPUT_DIR} --log_file_prefix="./resnet50_training_log_${PRECISION}""
    ARGS="$ARGS --train-no-eval --warmup-iterations 50 -b $BATCH_SIZE"
    LOG_PREFIX=resnet50_training_log
fi


if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    echo "running bf16 path"
elif [[ $PRECISION == "bf32" ]]; then
    ARGS="$ARGS --bf32"
    echo "running bf32 path"
elif [[ $PRECISION == "fp16" ]]; then
    ARGS="$ARGS --fp16"
    echo "running fp16 path"
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    echo "running fp32 path"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf16, bf32"
    exit 1
fi

ARGS="$ARGS -j 0 --seed 2020 --epochs $TRAINING_EPOCHS"

TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}

if [[ "0" == ${TORCH_INDUCTOR} ]];then
    if [[ -z "${DISTRIBUTED}" ]]; then
        ARGS="$ARGS --ipex"
        python -m intel_extension_for_pytorch.cpu.launch \
            ${ARGS_IPEX} \
            ${MODEL_DIR}/../../common/main.py \
            ${ARGS}
    else
        python -m intel_extension_for_pytorch.cpu.launch \
            ${ARGS_IPEX} \
            ${MODEL_DIR}/../../common/train.py \
            ${ARGS} 2>&1 | tee ${OUTPUT_DIR}/resnet50_dist_training_log_${PRECISION}.log
    fi
else
    export TORCHINDUCTOR_FREEZING=1
    ARGS="$ARGS --inductor"
    if [[ -z "${DISTRIBUTED}" ]]; then
        python -m intel_extension_for_pytorch.cpu.launch \
            ${ARGS_IPEX} \
            ${MODEL_DIR}/../../common/main.py \
            ${ARGS}
    else
        python -m intel_extension_for_pytorch.cpu.launch \
            ${ARGS_IPEX} \
            ${MODEL_DIR}/../../common/train.py \
            ${ARGS} 2>&1 | tee ${OUTPUT_DIR}/resnet50_dist_training_log_${PRECISION}.log
    fi
fi

wait

throughput=$(grep 'Training throughput:'  ${OUTPUT_DIR}/${LOG_PREFIX}_${PRECISION}_* |sed -e 's/.*throughput//;s/[^0-9.]//g' |awk '
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
echo "--------------------------------Performance Summary per NUMA Node--------------------------------"
echo "resnet50;"training throughput";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
latency="0.0"
accuracy="0.0"

echo "resnet50;"training throughput";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: examples per second
- key: latency
  value: $latency
  unit: seconds per example
- key: accuracy
  value: $accuracy
  unit: percentage
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
