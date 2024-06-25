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

if [ ! -e "${MODEL_DIR}/../../common/maskrcnn-benchmark/tools/train_net.py" ]; then
  echo "Could not find the script of train.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the train.py exist."
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

if [ -z  "${PRECISION}" ]; then
  echo "The PRECISION is not set"
  exit 1
fi

if [ -z  "${MODE}" ]; then
  echo "The MODE is not set"
  exit 1
fi

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ "$PRECISION" == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    echo "### running bf16 datatype"
elif [[ "$PRECISION" == "bf32" ]]; then
    ARGS="$ARGS --bf32"
    echo "### running bf32 datatype"
elif [[ "$PRECISION" == "fp32" || "$PRECISION" == "avx-fp32" ]]; then
    echo "### running fp32 datatype"
else
    echo "The specified precision '$PRECISION' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf16, and bf32."
    exit 1
fi

if [[ "$MODE" == "jit" ]]; then
    ARGS="$ARGS --jit"
    echo "### running jit mode"
elif [[ "$MODE" == "imperative" ]]; then
    echo "### running imperative mode"
else
    echo "The specified mode '$MODE' is unsupported."
    echo "Supported mode are: imperative and jit."
    exit 1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

export TRAIN=0

source "${MODEL_DIR}/../../common/utils.sh"
_get_platform_type

if [[ ${PLATFORM} == "windows" ]]; then
  CORES="${NUMBER_OF_PROCESSORS}"
else
  CORES=`lscpu | grep Core | awk '{print $4}'`
fi

if [ "$THROUGHPUT" ]; then 
    BATCH_SIZE=${BATCH_SIZE:-`expr $CORES \* 2`}
else 
    BATCH_SIZE=${BATCH_SIZE:-112}
fi

IPEX_ARGS=""
pip list | grep intel-extension-for-pytorch
if [[ "$?" == 0 ]]; then
  IPEX_ARGS="-m intel_extension_for_pytorch.cpu.launch \
    --enable_jemalloc --throughput_mode"
fi

if [ "$THROUGHPUT" ]; then 
    python ${IPEX_ARGS} \
    ${MODEL_DIR}/../../common/maskrcnn-benchmark/tools/test_net.py \
    $ARGS \
    --iter-warmup 10 \
    -i 20 \
    --config-file "${MODEL_DIR}/../../common/maskrcnn-benchmark/configs/e2e_mask_rcnn_R_50_FPN_1x_coco2017_inf.yaml" \
    TEST.IMS_PER_BATCH ${BATCH_SIZE} \
    MODEL.WEIGHT "${CHECKPOINT_DIR}/e2e_mask_rcnn_R_50_FPN_1x.pth" \
    MODEL.DEVICE cpu \
    2>&1 | tee ${OUTPUT_DIR}/maskrcnn_${PRECISION}_inference_throughput.log
    wait 

else 
    python ${IPEX_ARGS} \
    ${MODEL_DIR}/../../common/maskrcnn-benchmark/tools/test_net.py \
    $ARGS \
    --accuracy \
    --config-file "${MODEL_DIR}/../../common/maskrcnn-benchmark/configs/e2e_mask_rcnn_R_50_FPN_1x_coco2017_inf.yaml" \
    TEST.IMS_PER_BATCH ${BATCH_SIZE} \
    MODEL.WEIGHT "${CHECKPOINT_DIR}/e2e_mask_rcnn_R_50_FPN_1x.pth" \
    MODEL.DEVICE cpu \
    2>&1 | tee ${OUTPUT_DIR}/maskrcnn_${PRECISION}_accuracy.log
    wait
fi


if [[ ${PLATFORM} == "linux" ]]; then
  if [ "$THROUGHPUT" ]; then 
      LOG_0=${OUTPUT_DIR}/maskrcnn_${PRECISION}_inference_throughput*
  else 
      LOG_0=${OUTPUT_DIR}/maskrcnn_${PRECISION}_accuracy*
  fi

  throughput=$(grep 'Throughput:' ${LOG_0} |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
  latency=$(grep 'P99 Latency' ${LOG_0} | sed -e 's/.*P99 Latency//;s/[^0-9.]//g' | awk '
      BEGIN {
          sum = 0;
          i = 0;
      }
      {
          sum = sum + $1;
          i++;
      }
      END {
          sum = sum / i;
          printf("%.2f \n", sum);
      }')

  bbox_accuracy=$(grep 'bbox AP:' ${LOG_0} |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
  segm_accuracy=$(grep 'segm AP:' ${LOG_0} |sed -e 's/.*Accuracy//;s/[^0-9.]//g')

  echo ""maskrcnn";"throughput";$PRECISION;${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
  echo ""maskrcnn";"latency";$PRECISION;${BATCH_SIZE};${latency}" | tee -a ${OUTPUT_DIR}/summary.log
  echo ""maskrcnn";"bbox AP:";$PRECISION;${BATCH_SIZE};${bbox_accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
  echo ""maskrcnn";"segm AP:";$PRECISION;${BATCH_SIZE};${segm_accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
fi

yaml_content=$(cat << EOF
results: 
- key : throughput
  value: $throughput
  unit: fps
- key: latency
  value: $latency
  unit: ms
- key: bounding-box accuracy
  value: $bbox_accuracy
  unit: AP
- key: segmentation accuracy
  value: $segm_accuracy
  unit: AP
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
