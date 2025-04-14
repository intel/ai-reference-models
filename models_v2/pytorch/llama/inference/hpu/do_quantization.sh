#!/bin/bash

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


ARGS=""

FINETUNED_MODEL=${FINETUNED_MODEL:-"'meta-llama/Llama-2-7b-hf'"}
MODEL_HF=$(echo ${FINETUNED_MODEL} | cut -d'/' -f2 | tr -d "'")

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
#export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi
path="ipex"
ARGS="$ARGS --output_dir ${OUTPUT_DIR}  --lambada --jit"
echo "### running with intel extension for pytorch"


if [[ "$1" == "calibration" ]]
then
    precision="calibration"
    ARGS="$ARGS --dtype 'int8' --do-calibration --int8-qconfig '${OUTPUT_DIR}/${MODEL_HF}-qconfig.json' "
    echo "### running calibration to get qconfig"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported [calibration]"
    exit 1
fi

if [[ "$2" == "default" ]]
then
    ARGS="$ARGS --ipex_static_quantize "
    echo "### ipex_static_quantize"
elif [[ "$2" == "sq" ]]
then
    ARGS="$ARGS --ipex_smooth_quant "
    echo "###  ipex_smooth_quant"
else
    echo "The specified precision '$2' is unsupported."
    echo "Supported precisions are: default, sq"
    exit 1
fi

path=${3:-ipex}

if [[ "$path" == "ipex" ]]
then
    ARGS="$ARGS --ipex "
    echo "### do calibration for ipex"
elif [[ "$path" == "inductor" ]]
then
    ARGS="$ARGS --inductor "
    echo "### do calibration for inductor"
else
    echo "The specified backend '$3' is unsupported."
    echo "Supported Backends:[ipex, inductor]"
    exit 1
fi

mode="jit"
ARGS="$ARGS --jit --profile"
echo "### running with jit mode"


EVAL_SCRIPT=${EVAL_SCRIPT:-"${PWD}/run_llm.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}
rm -rf ${OUTPUT_DIR}/latency_log*
python -m intel_extension_for_pytorch.cpu.launch --memory-allocator tcmalloc --log_dir=${OUTPUT_DIR} --log_file_prefix="./latency_log_${precision}_${mode}" \
  ${EVAL_SCRIPT} $ARGS \
  --model-name-or-path   ${FINETUNED_MODEL} \
