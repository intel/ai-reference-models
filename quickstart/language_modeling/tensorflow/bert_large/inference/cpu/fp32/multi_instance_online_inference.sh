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

echo 'MODEL_DIR='$MODEL_DIR
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'DATASET_DIR='$DATASET_DIR

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

# Unzip pretrained model files
if [[ -z "${CHECKPOINT_DIR}" ]]; then
  pretrained_model_dir="pretrained_model/bert_large_checkpoints"
  if [ ! -d "${pretrained_model_dir}" ]; then
    unzip pretrained_model/bert_large_checkpoints.zip -d pretrained_model
  fi
  CHECKPOINT_DIR="${MODEL_DIR}/${pretrained_model_dir}"
fi

PRETRAINED_MODEL=${PRETRAINED_MODEL-${MODEL_DIR}/pretrained_model/fp32_bert_squad.pb}
CORES_PER_INSTANCE="4"

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

echo 'CHECKPOINT_DIR='$CHECKPOINT_DIR
echo 'PRETRAINED_MODEL='$PRETRAINED_MODEL

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name bert_large \
  --mode inference \
  --precision fp32 \
  --framework tensorflow \
  --batch-size ${BATCH_SIZE} \
  --data-location ${DATASET_DIR} \
  --in-graph ${PRETRAINED_MODEL} \
  --checkpoint ${CHECKPOINT_DIR} \
  --numa-cores-per-instance ${CORES_PER_INSTANCE} \
  --num-intra-threads 8 \
  --num-inter-threads 1 \
  --benchmark-only \
  --output-dir ${OUTPUT_DIR} \
  $@ \
  -- \
  init_checkpoint=model.ckpt-3649 \
  infer-option=SQuAD \
  experimental-gelu=True

if [[ $? == 0 ]]; then
  echo "Summary total images/sec:"
  grep 'throughput((num_processed_examples-threshold_examples)/Elapsedtime):' ${OUTPUT_DIR}/bert_large_fp32_inference_bs${BATCH_SIZE}_cores${CORES_PER_INSTANCE}_all_instances.log  | awk -F' ' '{sum+=$2;} END{print sum} '
else
  exit 1
fi
