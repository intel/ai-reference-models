#!/usr/bin/env bash
#
# Copyright (c) 2023 Intel Corporation
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
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist. Creating it..!"
  mkdir -p ${DATASET_DIR}
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, bfloat16 or fp16."
  exit 1
elif [ ${PRECISION} != "fp32" ] && [ ${PRECISION} != "bfloat16" ] && [ ${PRECISION} != "fp16" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32, bfloat16 and fp16"
  exit 1
fi

if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="32"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
export OMP_NUM_THREADS=${cores_per_socket}
NUM_INSTANCES="1"

source "${MODEL_DIR}/quickstart/common/utils.sh"
_ht_status_spr
_get_socket_cores_lists
_command numactl -C ${cores_per_socket_arr[0]} python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=gpt_j_6B \
  --precision=${PRECISION} \
  --mode=training \
  --framework=tensorflow \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --num-intra-threads ${cores_per_socket} \
  --num-inter-threads 2 \
  $@ \
  -- DEBIAN_FRONTEND=noninteractive \
  train-option=GLUE task-name=MRPC \
  do-train=True do-eval=True \
  install_transformer_fix=False \
  profile=False \
  learning-rate=2e-5 pad-to-max-length=True num-train-epochs=3 \
  cache-dir=${DATASET_DIR} output-dir=${OUTPUT_DIR} \
  warmup-steps=3 2>&1 | tee ${OUTPUT_DIR}/gpt_j_6B_glue_${PRECISION}_training_bs${BATCH_SIZE}_all_instances.log

if [[ $? == 0 ]]; then
  cat ${OUTPUT_DIR}/gpt_j_6B_glue_${PRECISION}_training_bs${BATCH_SIZE}_all_instances.log | grep "INFO:tensorflow:examples/sec:" | tail -n 2 | sed -e "s/.*: //"
  exit 0
else
  exit 1
fi
