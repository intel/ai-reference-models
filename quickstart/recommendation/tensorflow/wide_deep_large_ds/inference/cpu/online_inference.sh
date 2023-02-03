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

if [ ! -f "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32 or int8"
  exit 1
fi

if [[ $PRECISION != "fp32" ]] && [[ $PRECISION != "int8" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32 and int8"
  exit 1
fi

NUM_OMP_THREADS="${NUM_OMP_THREADS:-1}"

if [ -z "${PRETRAINED_MODEL}" ]; then
  if [[ $PRECISION == "fp32" ]]; then
    PRETRAINED_MODEL="${MODEL_DIR}/wide_deep_fp32_pretrained_model.pb"
  elif [[ $PRECISION == "int8" ]]; then
    PRETRAINED_MODEL="${MODEL_DIR}/wide_deep_int8_pretrained_model.pb"
  else
      echo "The specified precision '${PRECISION}' is unsupported."
      echo "Supported precisions are: fp32 and int8"
      exit 1
  fi
  if [[ ! -f "${PRETRAINED_MODEL}" ]]; then
    echo "The pretrained model could not be found. Please set the PRETRAINED_MODEL env var to point to the frozen graph file."
    exit 1
  fi
elif [[ ! -f "${PRETRAINED_MODEL}" ]]; then
  echo "The file specified by the PRETRAINED_MODEL environment variable (${PRETRAINED_MODEL}) does not exist."
  exit 1
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

# Run wide and deep large dataset inference
source "$MODEL_DIR/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
   --model-name wide_deep_large_ds \
   --precision ${PRECISION} \
   --mode inference \
   --framework tensorflow \
   --batch-size ${BATCH_SIZE} \
   --data-location $DATASET_DIR \
   --output-dir $OUTPUT_DIR \
   --in-graph ${PRETRAINED_MODEL} \
   --benchmark-only \
   $@ \
   -- num_omp_threads=$NUM_OMP_THREADS
