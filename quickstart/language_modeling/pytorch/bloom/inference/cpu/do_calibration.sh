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
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

path="ipex"
ARGS="$ARGS --use_ipex"
echo "### running with intel extension for pytorch"

precision="fp32"

CORES=`lscpu | grep Core | awk '{print $4}'`
BATCH_SIZE=${BATCH_SIZE:-1}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"Langboat/bloom-1b4-zh"}

EVAL_SCRIPT=${EVAL_SCRIPT:-"../../../../../../models/language_modeling/pytorch/bloom/run_clm.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}
rm -rf ${OUTPUT_DIR}/accuracy_log*
python -m intel_extension_for_pytorch.cpu.launch --ninstance 1 --node_id 0  --log_path=${OUTPUT_DIR} --log_file_prefix="accuracy_log_${precision}_${mode}" \
  ${EVAL_SCRIPT} $ARGS \
  --model_name_or_path ${PRETRAINED_MODEL} \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --do_eval \
  --do_calibration \
  --int8_config configure.json \
  --output_dir "./" \
  --calibration_iters 20 \
  --torch_dtype bfloat16 \
  --low_cpu_mem_usage
