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

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

# Training should have an empty output directory to prevent conflicts with previous weight files
if [[ "$(ls -A $OUTPUT_DIR)" ]]; then
  echo "The OUTPUT_DIR provided is not empty. Please provide an empty OUTPUT_DIR for training files."
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

# Check for precision
if [[ $PRECISION != "bfloat16" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Only bfloat16 precision is supported"
  exit 1
fi

export PYTHONPATH=$(pwd)/models/image_recognition/tensorflow/resnet50v1_5/training/gpu
export NUMBER_OF_PROCESS=2
export PROCESS_PER_NODE=2

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="256"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

source "$(dirname $0)/common/utils.sh"
_command mpirun -np $NUMBER_OF_PROCESS -ppn $PROCESS_PER_NODE --prepend-rank \
          python $PYTHONPATH/mlperf_resnet/imagenet_main.py 2 \
          --max_train_steps=1000 --train_epochs=1 --epochs_between_evals=1 \
          --inter_op_parallelism_threads 1 --intra_op_parallelism_threads 28  \
          --version 1 --resnet_size 50 \
          --data_dir=${DATASET_DIR} \
          --model_dir=${OUTPUT_DIR} \
          --use_synthetic_data --batch_size=$BATCH_SIZE --use_bfloat16 \
          --data_format=channels_last 2>&1 | tee ${OUTPUT_DIR}/resnet50_${PRECISION}_training_hvd.log
