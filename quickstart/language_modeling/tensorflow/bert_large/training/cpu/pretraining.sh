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

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32 or bfloat16."
  exit 1
fi

if [[ $PRECISION == "fp32" ]]; then
  BATCH_SIZE="32"
elif [[ $PRECISION == "bfloat16" ]]; then
  BATCH_SIZE="128"
else
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32 and bfloat16"
  exit 1
fi

cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
export OMP_NUM_THREADS=${cores_per_socket}
NUM_INSTANCES="2"

source "${MODEL_DIR}/quickstart/common/utils.sh"
_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=bert_large \
  --precision=${PRECISION} \
  --mode=training \
  --framework tensorflow \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --mpi_num_processes=${NUM_INSTANCES} \
  --mpi_num_processes_per_socket=1 \
  --batch-size ${BATCH_SIZE} \
  --num-intra-threads 64 \
  --num-inter-threads 1 \
  --num-train-steps=20 \
  $@ \
  -- DEBIAN_FRONTEND=noninteractive \
  train-option=Pretraining do-eval=False do-train=True profile=False \
  learning-rate=4e-5 max-predictions=76 max-seq-length=512 warmup-steps=0 \
  save-checkpoints_steps=1000 \
  config-file=${DATASET_DIR}/wwm_uncased_L-24_H-1024_A-16/bert_config.json \
  init-checkpoint=${DATASET_DIR}/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
  input-file=${DATASET_DIR}/tf_records/part-00430-of-00500 \
  experimental-gelu=True do-lower-case=False 2>&1 | tee ${OUTPUT_DIR}/bert_large_${PRECISION}_training_bs${BATCH_SIZE}_all_instances.log

if [[ $? == 0 ]]; then
  cat ${OUTPUT_DIR}/bert_large_${PRECISION}_training_bs${BATCH_SIZE}_all_instances.log | grep "INFO:tensorflow:examples/sec:" | tail -n 2 | sed -e "s/.*: //"
  exit 0
else
  exit 1
fi
