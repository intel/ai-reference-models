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
elif [ ${PRECISION} != "fp32" ] && [ ${PRECISION} != "bfloat16" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32 and bfloat16"
  exit 1
fi

if [ -z "${TF_MODELS_DIR}" ]; then
  echo "The required environment variable TF_MODELS_DIR has not been set."
  echo "Set TF_MODELS_DIR to the directory where the tensorflow/models repo has been cloned."
  exit 1
fi

if [ ! -d "${TF_MODELS_DIR}" ]; then
  echo "The TF_MODELS_DIR directory '${TF_MODELS_DIR}' does not exist"
  exit 1
fi

# Apply the TF 2.0 patch to the TF_MODELS_DIR
cd ${TF_MODELS_DIR}
if [ ${PRECISION} == "fp32" ]; then
  git apply ${MODEL_DIR}/models/object_detection/tensorflow/ssd-resnet34/training/fp32/tf-2.0.diff
elif [ ${PRECISION} == "bfloat16" ]; then
  git apply ${MODEL_DIR}/models/object_detection/tensorflow/ssd-resnet34/training/bfloat16/tf-2.0.diff
fi
cd ${MODEL_DIR}

# Get number of cores per socket line from lscpu
cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
cores_per_socket="${cores_per_socket//[[:blank:]]/}"

# Subtract 4 to use as the num_intra_threads
num_intra_threads=$(($cores_per_socket - 4))

BATCH_SIZE="56"
NUM_INSTANCES="2"

source "${MODEL_DIR}/quickstart/common/utils.sh"
_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=ssd-resnet34 \
  --precision=${PRECISION} \
  --mode=training \
  --framework tensorflow \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --model-source-dir ${TF_MODELS_DIR} \
  --mpi_num_processes=${NUM_INSTANCES} \
  --mpi_num_processes_per_socket=1 \
  --batch-size ${BATCH_SIZE} \
  --num-intra-threads ${num_intra_threads} \
  --num-inter-threads 1 \
  --num-cores ${cores_per_socket} \
  --num-train-steps 100  --num_warmup_batches=20  --weight_decay=1e-4 \
  $@ 2>&1 | tee ${OUTPUT_DIR}/ssd_resnet34_${PRECISION}_training_bs${BATCH_SIZE}_all_instances.log

if [[ $? == 0 ]]; then
  cat ${OUTPUT_DIR}/ssd_resnet34_${PRECISION}_training_bs${BATCH_SIZE}_all_instances.log | grep "total images/sec:" | sed -e s"/.*: //"
  exit 0
else
  exit 1
fi
