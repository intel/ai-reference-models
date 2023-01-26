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
if [ ! -e "${MODEL_DIR}/models/language_modeling/pytorch/t5/inference/cpu/run_translation.py"  ]; then
    echo "Could not find the script of main.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the main.py exist at the: \${MODEL_DIR}/models/language_modeling/pytorch/t5/inference/cpu/run_translation.py"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, int8."
  exit 1
fi

if [ -z "${MODEL_NAME}" ]; then
  echo "The required environment variable MODEL_NAME has not been set"
  echo "Supported MODEL_NAME are: t5-small, t5-base, t5-large, t5-3b and t5-11b"
  exit 1
fi

if [ -z "${MAX_PREDICT_SAMPLES}" ]; then
  echo "The required environment variable MAX_PREDICT_SAMPLES has not been set"
  exit 1
fi

if [ -z "${CORES_PER_INSTANCE}" ]; then
  echo "The required environment variable CORES_PER_INSTANCE has not been set"
  exit 1
fi

BATCH_SIZE=1

ARGS=""
IPEX_ARGS=""

if [[ $PRECISION == "int8" ]]; then
    echo "running int8 path"
    ARGS="$ARGS --do_quantization 1"
elif [[ $PRECISION == "fp32" ]]; then
    echo "running fp32 path"
    ARGS="$ARGS --do_quantization 0"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, int8"
    exit 1
fi

OPTIMIZATION="pytorch"
if [[ "$1" == "ipex" ]]
then
    OPTIMIZATION="ipex"
    ARGS="$ARGS --do_ipex_optimization 1"
    echo "### running ipex optimization path"
else
    ARGS="$ARGS --do_ipex_optimization 0"
    echo "### running offical PyTorch path"
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

BATCH_SIZE=1

rm -rf ${OUTPUT_DIR}/${MODEL_NAME}_log*
SOURCE_PREFIX="translate English to Romanian: "

#Get platform type
source "${MODEL_DIR}/quickstart/common/utils.sh"
_get_platform_type

#Add platform specific source_prefix
if [[ ${PLATFORM} == "windows" ]]
then
	SOURCE_PREFIX='translate English to Romanian:'
else
	SOURCE_PREFIX=\"translate English to Romanian: \"
fi

# check if stoch PYT or IPEX is installed on the system
pip list | grep intel-extension-for-pytorch
if [[ "$?" == 0 ]]; then
  IPEX_ARGS="-m intel_extension_for_pytorch.cpu.launch \
	  --use_default_allocator \
	  --ninstances 1 \
    --ncore_per_instance ${CORES_PER_INSTANCE} \
    --log_path=${OUTPUT_DIR} \
    --log_file_prefix="./${MODEL_NAME}_log_${PRECISION}_${OPTIMIZATION}""
fi

python ${IPEX_ARGS} \
    ${MODEL_DIR}/models/language_modeling/pytorch/t5/inference/cpu/run_translation.py \
    $ARGS \
    --model_name_or_path $MODEL_NAME \
    --do_predict \
    --max_predict_samples $MAX_PREDICT_SAMPLES \
    --overwrite_cache 1 \
    --source_lang en \
    --target_lang ro \
    --source_prefix ${SOURCE_PREFIX} \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir ${OUTPUT_DIR} \
    --per_device_eval_batch_size $BATCH_SIZE \
    --overwrite_output_dir \
    --predict_with_generate

#For the summary of results
wait

if [[ ${PLATFORM} != "windows" ]]; then
  key_words="predict_samples_per_second"

  METRIC=$(grep ${key_words} ${OUTPUT_DIR}/${MODEL_NAME}_log_${PRECISION}_${OPTIMIZATION}*.log | awk '{print $4}')
  echo $METRIC
  echo "${MODEL_NAME};"Inference performance";${PRECISION};${BATCH_SIZE};${OPTIMIZATION};${METRIC}" | tee -a ${OUTPUT_DIR}/summary.log
fi
