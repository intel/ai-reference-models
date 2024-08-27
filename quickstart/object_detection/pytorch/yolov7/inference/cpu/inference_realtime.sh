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
if [ ! -e "${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7_ipex.patch"  ]; then
    echo "Could not find the script of yolov7_ipex.patch. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the yolov7_ipex.patch exist at the: \${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7_ipex.patch"
    exit 1
else
    TMP_PATH=$(pwd)
    cd "${MODEL_DIR}/models/object_detection/pytorch/yolov7/"
    if [ ! -d "yolov7" ]; then
        git clone https://github.com/WongKinYiu/yolov7.git yolov7
        cd yolov7
        cp ../yolov7_int8_default_qparams.json .
        cp ../yolov7.py .
        pip install -r requirements.txt
        git checkout a207844
        git apply ../yolov7_ipex.patch
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

BATCH_SIZE=1

rm -rf ${OUTPUT_DIR}/yolov7_latency_log*

cd "${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7/"

ARGS="--checkpoint-dir $CHECKPOINT_DIR --weights yolov7.pt"
ARGS="$ARGS --img 640 -e --performance --data data/coco.yaml --dataset-dir $DATASET_DIR --conf-thres 0.001 --iou 0.65 --device cpu"

if [[ $PRECISION == "int8" ]]; then
    echo "running int8 path"
    ARGS="$ARGS --int8 --configure-dir ${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7/yolov7_int8_default_qparams.json"
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
    echo "Supported precisions are: fp32, bf16, int8, and bf32"
    exit 1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
CORES_PER_NUMA=`expr $CORES \* $SOCKETS / $NUMAS`

CORES_PER_INSTANCE=4

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=$CORES_PER_INSTANCE

NUMBER_INSTANCE=`expr $CORES_PER_NUMA / $CORES_PER_INSTANCE`

TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}
if [[ "0" == ${TORCH_INDUCTOR} ]];then
    python -m intel_extension_for_pytorch.cpu.launch \
        --memory-allocator jemalloc \
        --ninstances $NUMAS \
        --log_path=${OUTPUT_DIR} \
        --log_file_prefix="./yolov7_latency_log_${PRECISION}" \
        ${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7/yolov7.py \
        $ARGS \
        --ipex \
        --batch-size $BATCH_SIZE \
        --weight-sharing \
        --number-instance $NUMBER_INSTANCE
else
    echo "Running yolov7 inference with torch.compile inductor backend."
    export TORCHINDUCTOR_FREEZING=1
    python -m intel_extension_for_pytorch.cpu.launch \
        --memory-allocator jemalloc \
        --ninstance ${NUMAS} \
        --log_path=${OUTPUT_DIR} \
        --log_file_prefix="./yolov7_latency_log_${PRECISION}" \
        ${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7/yolov7.py \
        $ARGS \
        --inductor \
        --batch-size $BATCH_SIZE \
	    --weight-sharing \
        --number-instance $NUMBER_INSTANCE
fi

wait
cd -

TOTAL_CORES=`expr $CORES \* $SOCKETS`
INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`

throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/yolov7_latency_log* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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
        printf("%.2f", sum);
}')

p99_latency=$(grep 'P99 Latency' ${OUTPUT_DIR}/yolov7_latency_log* |sed -e 's/.*P99 Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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
echo "yolov7;"latency";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
echo "yolov7;"p99_latency";${PRECISION};${BATCH_SIZE};${p99_latency}" | tee -a ${OUTPUT_DIR}/summary.log
