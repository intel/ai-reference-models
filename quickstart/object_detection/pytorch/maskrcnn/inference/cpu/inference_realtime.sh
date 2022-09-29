#!/usr/bin/env bash
#
# Copyright (c) 2020 Intel Corporation
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

if [ ! -e "${MODEL_DIR}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/tools/test_net.py" ]; then
  echo "Could not find the script test_net.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the test_net.py exist at: \${MODEL_DIR}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/tools/test_net.py"
  exit 1
fi

if [ ! -e "${CHECKPOINT_DIR}/e2e_mask_rcnn_R_50_FPN_1x.pth" ]; then
  echo "The pretrained model \${CHECKPOINT_DIR}/e2e_mask_rcnn_R_50_FPN_1x.pth does not exist"
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

if [[ "$1" == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    echo "### running bf16 datatype"
elif [[ "$1" == "bf32" ]]; then
    ARGS="$ARGS --bf32"
    echo "### running bf32 datatype"
elif [[ "$1" == "fp32" || "$1" == "avx-fp32" ]]; then
    echo "### running fp32 datatype"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf16, and bf32."
    exit 1
fi

if [[ "$2" == "jit" ]]; then
    ARGS="$ARGS --jit"
    echo "### running jit mode"
elif [[ "$2" == "imperative" ]]; then
    echo "### running imperative mode"
else
    echo "The specified mode '$2' is unsupported."
    echo "Supported mode are: imperative and jit."
    exit 1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

export TRAIN=0

PRECISION=$1
BATCH_SIZE=1

rm -rf ${OUTPUT_DIR}/maskrcnn_${PRECISION}_inference_realtime*

# check if stoch PYT or IPEX is installed on the system
IPEX_ARGS=""
pip list | grep intel-extension-for-pytorch
if [[ "$?" == 0 ]]; then
  IPEX_ARGS="-m intel_extension_for_pytorch.cpu.launch \
    --enable_jemalloc --latency_mode"
fi

python ${IPEX_ARGS} \
    ${MODEL_DIR}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/tools/test_net.py \
    $ARGS \
    --iter-warmup 20 \
    -i 200 \
    --config-file "${MODEL_DIR}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/configs/e2e_mask_rcnn_R_50_FPN_1x_coco2017_inf.yaml" \
    TEST.IMS_PER_BATCH ${BATCH_SIZE} \
    MODEL.WEIGHT "${CHECKPOINT_DIR}/e2e_mask_rcnn_R_50_FPN_1x.pth" \
    MODEL.DEVICE cpu \
    2>&1 | tee ${OUTPUT_DIR}/maskrcnn_${PRECISION}_inference_realtime.log

# For the summary of results
wait

source "${MODEL_DIR}/quickstart/common/utils.sh"
_get_platform_type

if [[ ${PLATFORM} == "linux" ]]; then
  CORES=`lscpu | grep Core | awk '{print $4}'`
  CORES_PER_INSTANCE=4

  INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET=`expr $CORES / $CORES_PER_INSTANCE`

  throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/maskrcnn_${PRECISION}_inference_realtime* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET '
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
  p99_latency=$(grep 'P99 Latency' ${OUTPUT_DIR}/maskrcnn_${PRECISION}_inference_realtime* |sed -e 's/.*P99 Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET '
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
    printf("%.3f ms", sum);
}')
  echo ""maskrcnn";"latency";$1;${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
  echo ""maskrcnn";"p99_latency";$1;${BATCH_SIZE};${p99_latency}" | tee -a ${OUTPUT_DIR}/summary.log
fi
