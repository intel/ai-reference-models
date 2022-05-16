#!/usr/bin/env bash
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

MODEL_DIR=${MODEL_DIR-$PWD}
if [ ! -e "${MODEL_DIR}/models/image_recognition/pytorch/common/main.py"  ]; then
    echo "Could not find the script of main.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the main.py exist at the: \${MODEL_DIR}/models/image_recognition/pytorch/common/main.py"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
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
  echo "Please set PRECISION to fp32, avx-fp32, int8, avx-int8, or bf16."
  exit 1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

BATCH_SIZE=128

rm -rf ${OUTPUT_DIR}/resnet50_accuracy_log*

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

# download pretrained weight.
python ${MODEL_DIR}/models/image_recognition/pytorch/common/hub_help.py \
    --url https://download.pytorch.org/models/resnet50-0676ba61.pth

ARGS=""
ARGS="$ARGS -e -a resnet50 ${DATASET_DIR}"

if [[ $PRECISION == "int8" || $PRECISION == "avx-int8" ]]; then
    echo "running int8 path"
    ARGS="$ARGS --int8 --configure-dir ${MODEL_DIR}/models/image_recognition/pytorch/common/resnet50_configure_sym.json"
elif [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16 --jit"
    echo "running bf16 path"
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    ARGS="$ARGS --jit"
    echo "running fp32 path"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf16, int8, and avx-int8"
    exit 1
fi

weight_sharing=false
if [ ${WEIGHT_SHAREING} ]; then
  echo "Running RN50 inference throughput with runtime extension enabled."
  weight_sharing=true
fi

if [ "$weight_sharing" = true ]; then
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    CORES_PER_INSTANCE=$CORES
    INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
    LAST_INSTANCE=`expr $INSTANCES - 1`
    INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`

    BATCH_PER_STREAM=2
    CORES_PER_STREAM=1
    STREAM_PER_INSTANCE=`expr $CORES / $CORES_PER_STREAM`
    BATCH_SIZE=`expr $BATCH_PER_STREAM \* $STREAM_PER_INSTANCE`

    export OMP_NUM_THREADS=$CORES_PER_STREAM

    numa_node_i=0
    start_core_i=0
    end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
    LOG_i=resnet50_accuracy_log_${PRECISION}_0.log
    echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
    numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python -u \
        ${MODEL_DIR}/models/image_recognition/pytorch/common/main_runtime_extension.py \
        $ARGS \
        --ipex \
        --pretrained \
        -j 0 \
        -b $BATCH_SIZE \
        --number-instance $STREAM_PER_INSTANCE \
        --use-multi-stream-module \
        --instance-number 0

else

  python -m intel_extension_for_pytorch.cpu.launch \
      --use_default_allocator \
      ${MODEL_DIR}/models/image_recognition/pytorch/common/main.py \
      $ARGS \
      --ipex \
      --pretrained \
      -j 0 \
      -b $BATCH_SIZE 2>&1 | tee ${OUTPUT_DIR}/resnet50_accuracy_log_${PRECISION}.log
fi
wait

accuracy=$(grep 'Accuracy:' ${OUTPUT_DIR}/resnet50_accuracy_log_${PRECISION}.log |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
echo "resnet50;"accuracy";${PRECISION};${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
