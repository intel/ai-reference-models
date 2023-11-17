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

if [ ! -e "${MODEL_DIR}/models/diffusion/pytorch/stable_diffusion/textual_inversion.py"  ]; then
    echo "Could not find the script of textual_inversion.py. Please set environment variable '\${MODEL_DIR}'."
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

ARGS=""
if [ "$1" == "bf16" ]; then
    ARGS="$ARGS --precision=bf16"
    echo "### running bf16 datatype"
elif [ "$1" == "fp16" ]; then
    ARGS="$ARGS --precision=fp16"
    echo "### running fp16 datatype"
elif [ "$1" == "bf32" ]; then
    ARGS="$ARGS --precision=bf32"
    echo "### running bf32 datatype"
elif [ "$1" == "fp32" ]; then
    ARGS="$ARGS --precision=fp32"
    echo "### running fp32 datatype"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, bf32, fp16, bf16"
    exit 1
fi


CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

CORES_PER_INSTANCE=$CORES

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

PRECISION=$1

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export DATA_DIR="./cat"

rm -rf ${OUTPUT_DIR}/stable_diffusion_finetune_log_${PRECISION}*

python -m intel_extension_for_pytorch.cpu.launch \
    --memory-allocator tcmalloc \
    --ninstances 1 \
    --ncores_per_instance ${CORES_PER_INSTANCE} \
    --log_dir=${OUTPUT_DIR} \
    --log_file_prefix="./stable_diffusion_finetune_log_${PRECISION}" \
    ${MODEL_DIR}/models/diffusion/pytorch/stable_diffusion/textual_inversion.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATA_DIR \
    --learnable_property="object" \
    --placeholder_token="\"<cat-toy>\"" --initializer_token="toy" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    -w 10 --max_train_steps=20 \
    --learning_rate=2.0e-03 --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --train-no-eval \
    --ipex $ARGS

# For the summary of results
wait

throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/stable_diffusion_finetune_log_${PRECISION}* |sed -e 's/.*Throughput//;s/[^0-9.]//g')
echo ""stable_diffusion";"finetune";"throughput";$1;${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
