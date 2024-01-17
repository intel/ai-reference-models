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

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
#export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

ARGS="$ARGS --accuracy_only  --lambada"
echo "### running with intel extension for pytorch"

if [[ "$1" == "fp32" ]]
then
    precision="fp32"
    ARGS="$ARGS --dtype 'fp32' "
    echo "### running fp32 mode"
elif [[ "$1" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --dtype 'bf16' "
    echo "### running bf16 mode"
elif [[ "$1" == "fp32" ]]
then
    echo "### running fp32 mode"
elif [[ "$1" == "fp16" ]]
then
    precision=fp16
    ARGS="$ARGS --dtype 'fp16'"
    echo "### running fp16 mode"
elif [[ "$1" == "bf32" ]]
then
    precision="bf32"
    ARGS="$ARGS --dtype 'bf32'"
    echo "### running bf32 mode"
elif [[ "$1" == "int8-fp32" ]]
then
    precision="int8-fp32"
    ARGS="$ARGS --dtype 'int8' --int8-qconfig '${OUTPUT_DIR}/qconfig.json'"
    echo "### running int8-fp32 mode"
elif [[ "$1" == "int8-bf16" ]]
then
    precision="int8-bf16"
    ARGS="$ARGS --dtype 'int8' --int8_bf16_mixed --int8-qconfig '${OUTPUT_DIR}/qconfig.json'"
    echo "### running int8-bf16 mode"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, bf32, bf16, fp16, int8-fp32, int8-bf16"
    exit 1
fi


if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi

FINETUNED_MODEL=${FINETUNED_MODEL:-"'THUDM/chatglm3-6b'"}

EVAL_SCRIPT=${EVAL_SCRIPT:-"../../../../../../models/language_modeling/pytorch/chatglm/inference/cpu/run_llm.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}
rm -rf ${OUTPUT_DIR}/latency_log*

TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}
if [[ "0" == ${TORCH_INDUCTOR} ]];then
    path="ipex"
    mode="jit"
    ARGS="$ARGS --jit"
    echo "### running with jit mode"
    if [[ "$1" == "int8-bf16" || "$1" == "int8-fp32" ]];then
        ARGS="$ARGS --ipex_smooth_quant"
    fi
    python -m intel_extension_for_pytorch.cpu.launch --node_id 0 --enable_tcmalloc --log_path=${OUTPUT_DIR} --log_file_prefix="./ChatGLM_${precision}_accuracy_${mode}" \
        ${EVAL_SCRIPT} $ARGS \
        --ipex \
        --model-name-or-path   ${FINETUNED_MODEL}
else
    echo "### running with torch.compile inductor backend"
    export TORCHINDUCTOR_FREEZING=1
    python -m intel_extension_for_pytorch.cpu.launch --node_id 0 --enable_tcmalloc --log_path=${OUTPUT_DIR} --log_file_prefix="./ChatGLM_${precision}_accuracy_${mode}" \
        ${EVAL_SCRIPT} $ARGS \
        --inductor \
        --model-name-or-path   ${FINETUNED_MODEL}
fi

accuracy=$(cat ${OUTPUT_DIR}/ChatGLM_${precision}_accuracy* | grep "Accuracy:" |sed -e 's/.*= //;s/[^0-9.]//g')

echo ""ChatGLM";"accuracy";${precision};${BATCH_SIZE};${accuracy}" | tee -a ${WORK_SPACE}/summary.log
