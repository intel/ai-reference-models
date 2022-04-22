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
ARGS="$ARGS --use_ipex "
echo "### running with intel extension for pytorch calibration"

BATCH_SIZE=${BATCH_SIZE:-8}
FINETUNED_MODEL=${FINETUNED_MODEL:-"distilbert-base-uncased-distilled-squad"}
if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi
EVAL_SCRIPT=${EVAL_SCRIPT:-"./transformers/examples/pytorch/question-answering/run_qa.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}


python -m intel_extension_for_pytorch.cpu.launch --ninstance 1 --node_id 0 --enable_jemalloc \
  ${EVAL_SCRIPT} $ARGS \
  --model_name_or_path   ${FINETUNED_MODEL} \
  --dataset_name squad \
  --do_eval \
  --max_seq_length 128 \
  --doc_stride 64 \
  --output_dir ./tmp \
  --per_device_eval_batch_size $BATCH_SIZE \
  --do_calibration \
  --int8_config configure.json