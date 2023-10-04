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
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${INIT_CHECKPOINT_DIR}" ]; then
  echo "The required environment variable INIT_CHECKPOINT_DIR has not been set"
  exit 1
fi

if [ ! -d "${INIT_CHECKPOINT_DIR}" ]; then
  echo "The INIT_CHECKPOINT_DIR '${INIT_CHECKPOINT_DIR}' does not exist"
  exit 1
fi


# If precision env is not mentioned, then the workload will run with the default precision.
if [ -z "${PRECISION}"]; then
  PRECISION=fp32
  echo "Running with default precision ${PRECISION}"
fi

if [[ $PRECISION != "fp32" ]] && [[ $PRECISION != "bfloat16" ]] && [[ $PRECISION != "fp16" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precision is float32 , bfloat16, float16."
  exit 1
fi

# If steps env is not mentioned, then the workload will run with the default steps.
if [ -z "${EPOCHS}"]; then
  EPOCHS="0"
  echo "Warning : No. of Epochs not set, it will run epochs required for 30k steps for given batch size"
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="512"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

# Get number of cores per socket line from lscpu
cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
cores_per_socket="${cores_per_socket//[[:blank:]]/}"

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
CORES_PER_NUMA=`expr $CORES \* $SOCKETS / $NUMAS`

NUM_INSTANCES=`expr $cores_per_socket / $CORES_PER_NUMA`

echo "Running multi-instance vision transformer fine-tuning"

# Run vision transformer training
source "$MODEL_DIR/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
   --model-name vision_transformer \
   --precision ${PRECISION} \
   --mode training  \
   --framework tensorflow \
   --batch-size ${BATCH_SIZE} \
   --mpi_num_processes=${NUM_INSTANCES} \
   --mpi_num_processes_per_socket=${NUM_INSTANCES} \
   --num-intra-threads $CORES_PER_NUMA \
   --num-inter-threads 2 \
   --epochs=${EPOCHS} \
   --data-location $DATASET_DIR \
   --checkpoint $OUTPUT_DIR \
   $@ \
   --init-checkpoint=$INIT_CHECKPOINT_DIR


