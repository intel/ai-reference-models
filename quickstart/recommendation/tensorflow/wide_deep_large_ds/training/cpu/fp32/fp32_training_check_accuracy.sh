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

# This script trains the model for a specified number of steps, then compares
# the accuracy that is printed in the log file against a target accuracy. If
# the evaluated accuracy is less than the target, the script will fail.
#
# Define the target accuracy using the TARGET_ACCURACY environment variable
# (for example TARGET_ACCURACY=0.75). Set the number of steps to run using the
# STEPS environment variable (this will default to run 500 steps, if the
# variable has not been set).

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

if [ ! -z "${TARGET_ACCURACY}" ]; then
  echo "Target Accuracy: ${TARGET_ACCURACY}"
fi

# Define a checkpoint arg, if CHECKPOINT_DIR was provided
CHECKPOINT_ARG=""
if [ ! -z "${CHECKPOINT_DIR}" ]; then
  # If a checkpoint dir was provided, ensure that it exists, then setup the arg
  mkdir -p ${CHECKPOINT_DIR}
  CHECKPOINT_ARG="--checkpoint=${CHECKPOINT_DIR}"
fi

# Set default number of steps
STEPS=${STEPS:-500}

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="512"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

# Run wide and deep large ds training while saving output to a temp file
temp_output_file=$(mktemp /tmp/output.XXXXXXXXX)
source "$MODEL_DIR/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
 --model-name wide_deep_large_ds \
 --precision fp32 \
 --mode training  \
 --framework tensorflow \
 --batch-size ${BATCH_SIZE} \
 --data-location $DATASET_DIR \
 $CHECKPOINT_ARG \
 --output-dir $OUTPUT_DIR \
 $@ \
 -- steps=$STEPS 2>&1 | tee $temp_output_file

# If a target accuracy has been specified, find the accuracy from the log file
if [ ! -z "${TARGET_ACCURACY}" ]; then
  # Get the accuracy output from the training script
  accuracy_output=$(grep 'Accuracy:' $temp_output_file)
  accuracy_output=$(echo $accuracy_output | sed 's/^Accuracy: //')
fi

# Remove the temp output file
if [[ -f $temp_output_file ]]; then
  rm $temp_output_file
fi

# If a target accuracy has been specified, compare the actual accuracy to the
# target, and fail if the accuracy target has not been met
if [ ! -z "${TARGET_ACCURACY}" ]; then
  if [[ -z $accuracy_output ]]; then
    echo "Unable to find accuracy metric in the log file"
    exit 1
  fi

  # Compare the accuracy to the target accuracy - 1 means the accuracy target was met
  accuracy_met=$(python -c "print(int($accuracy_output >= $TARGET_ACCURACY))")
  if [[ $accuracy_met == 1 ]]; then
    echo "Accuracy of $accuracy_output met the target accuracy of $TARGET_ACCURACY"
    exit 0
  else
    echo "Accuracy of $accuracy_output did not meet the target accuracy of $TARGET_ACCURACY"
    exit 1
  fi
fi
