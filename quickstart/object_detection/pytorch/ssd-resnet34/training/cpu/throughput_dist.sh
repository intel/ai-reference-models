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

if [ ! -e "${MODEL_DIR}/models/object_detection/pytorch/ssd-resnet34/training/cpu/train.py" ]; then
  echo "Could not find the script of train.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the train.py exist at the: \${MODEL_DIR}/models/object_detection/pytorch/ssd-resnet34/training/cpu/train.py"
  exit 1
fi

if [ ! -e "${CHECKPOINT_DIR}/ssd/resnet34-333f7ec4.pth" ]; then
  echo "The pretrained model \${CHECKPOINT_DIR}/ssd/resnet34-333f7ec4.pth does not exist"
  exit 1
fi

if [ ! -d "${DATASET_DIR}/coco" ]; then
  echo "The DATASET_DIR \${DATASET_DIR}/coco does not exist"
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
    ARGS="$ARGS --autocast"
    echo "### running bf16 datatype"
elif [[ $1 == "fp32" || $1 == "avx-fp32" ]]; then
    echo "### running fp32 datatype"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, and bf16"
    exit 1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
NNODES=${NNODES:-1}
HOSTFILE=${HOSTFILE:-./hostfile}
NUM_RANKS=$(( NNODES * SOCKETS ))

CORES_PER_INSTANCE=$CORES

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export USE_IPEX=1
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

PRECISION=$1
BATCH_SIZE=100

rm -rf ${OUTPUT_DIR}/train_ssdresnet34_${PRECISION}_throughput_dist*

torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh

python -m intel_extension_for_pytorch.cpu.launch \
    --use_default_allocator \
    --ncore_per_instance ${CORES_PER_INSTANCE} \
    --distributed \
    --nnodes ${NNODES} \
    --hostfile ${HOSTFILE} \
    --nproc_per_node ${SOCKETS} \
    ${MODEL_DIR}/models/object_detection/pytorch/ssd-resnet34/training/cpu/train.py \
    --epochs 70 \
    --warmup-factor 0 \
    --lr 2.5e-3 \
    --threshold=0.23 \
    --seed 2000 \
    --log-interval 10 \
    --data ${DATASET_DIR}/coco \
    --batch-size ${BATCH_SIZE} \
    --pretrained-backbone ${CHECKPOINT_DIR}/ssd/resnet34-333f7ec4.pth \
    --performance_only \
    -w 20 \
    -iter 1000 \
    --world_size ${NUM_RANKS} \
    --backend ccl \
    $ARGS 2>&1 | tee ${OUTPUT_DIR}/train_ssdresnet34_${PRECISION}_throughput_dist.log

# For the summary of results
wait

throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/train_ssdresnet34_${PRECISION}_throughput_dist* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
echo ""SSD-RN34";"training distributed throughput";$1; ${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
