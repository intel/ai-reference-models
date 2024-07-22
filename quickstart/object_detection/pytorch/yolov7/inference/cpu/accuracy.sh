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
if [ ! -d "${DATASET_DIR}/coco" ]; then
  echo "The DATASET_DIR \${DATASET_DIR}/coco does not exist"
  exit 1
fi

if [ ! -e "${CHECKPOINT_DIR}/yolov7.pt" ]; then
  echo "The CHECKPOINT_DIR \${CHECKPOINT_DIR}/yolov7.pt does not exist"
  exit 1
fi

cd $DATASET_DIR
DATASET_DIR=$(pwd)
cd -

cd $CHECKPOINT_DIR
CHECKPOINT_DIR=$(pwd)
cd -

MODEL_DIR=${MODEL_DIR-$PWD}
if [ ! -e "${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7_ipex_and_inductor.patch"  ]; then
    echo "Could not find the script of yolov7_ipex_and_inductor.patch. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the yolov7_ipex_and_inductor.patch exist at the: \${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7_ipex_and_inductor.patch"
    exit 1
else
    TMP_PATH=$(pwd)
    cd "${MODEL_DIR}/models/object_detection/pytorch/yolov7/"
    if [ ! -d "yolov7" ]; then
        git clone https://github.com/WongKinYiu/yolov7.git yolov7
        cd yolov7
        cp ../yolov7.py .
        pip install -r requirements.txt
        git checkout a207844
        git apply ../yolov7_ipex_and_inductor.patch
    else
        cd yolov7
    fi
    cd $TMP_PATH
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

cd ${OUTPUT_DIR}
OUTPUT_DIR=$(pwd)
cd -

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to int8, fp32, bf32, bf16, or fp16."
  exit 1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

BATCH_SIZE=${BATCH_SIZE:-40}

rm -rf ${OUTPUT_DIR}/yolov7_accuracy_log*

cd "${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7"

ARGS="--checkpoint-dir $CHECKPOINT_DIR --weights yolov7.pt"
ARGS="$ARGS --img 640 -e --data data/coco.yaml --dataset-dir $DATASET_DIR --conf-thres 0.001 --iou 0.65 --device cpu"

if [[ $PRECISION == "int8" ]]; then
    echo "running int8 path"
    ARGS="$ARGS --int8"
elif [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16 --jit"
    echo "running bf16 path"
elif [[ $PRECISION == "bf32" ]]; then
    ARGS="$ARGS --bf32 --jit"
    echo "running bf32 path"
elif [[ $PRECISION == "fp16" ]]; then
    ARGS="$ARGS --fp16 --jit"
    echo "running fp16 path"
elif [[ $PRECISION == "fp32" ]]; then
    ARGS="$ARGS --jit"
    echo "running fp32 path"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, bf16, bf32, int8"
    exit 1
fi


TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}

if [[ "0" == ${TORCH_INDUCTOR} ]];then
    python -m intel_extension_for_pytorch.cpu.launch \
      --memory-allocator jemalloc \
      --skip-cross-node-cores \
      --log_dir=${OUTPUT_DIR} \
      --log_file_prefix="./yolov7_accuracy_log_${PRECISION}" \
      ${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7/yolov7.py \
      $ARGS \
      --ipex \
      --batch-size $BATCH_SIZE
else
    echo "Running yolov7 inference with torch.compile inductor backend."
    export TORCHINDUCTOR_FREEZING=1
    python -m torch.backends.xeon.run_cpu \
      --enable-jemalloc \
      --skip-cross-node-cores \
      --log_path=${OUTPUT_DIR} \
      ${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7/yolov7.py \
      $ARGS \
      --inductor \
      --batch-size $BATCH_SIZE
fi
wait
cd -

accuracy=$(grep -F 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' \
    ${OUTPUT_DIR}/yolov7_accuracy_log_${PRECISION}_*.log | \
    awk -F '=' '{print $NF}')
echo "yolov7;"accuracy";${PRECISION};${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
