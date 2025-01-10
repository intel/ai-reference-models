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
MODE="inference"

# echo 'MODEL_DIR='$MODEL_DIR
#echo 'OUTPUT_DIR='$OUTPUT_DIR
#echo 'DATASET_DIR='$DATASET_DIR

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
elif [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [[ -z "${PRECISION}" ]]; then
  PRECISION=fp32
  echo "Running with default precision ${PRECISION}"
fi

if [[ $PRECISION != "fp32" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precision is fp32."
  exit 1
fi

if [ -z "${PRETRAINED_MODEL}" ]; then
  echo "Please set the PRETRAINED_MODEL environment variable to point to the directory containing the pretrained model."
  exit 1
elif [[ ! -d "${PRETRAINED_MODEL}" ]]; then
  echo "The directory specified by the PRETRAINED_MODEL environment variable (${PRETRAINED_MODEL}) does not exist."
  exit 1
fi

# Create an array of input directories that are expected and then verify that they exist
if [[ -z "${BATCH_SIZE}" ]]; then
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

# If cores per instance env is not mentioned, then the workload will run with the default value.
if [ -z "${CORES_PER_INSTANCE}" ]; then
  CORES_PER_INSTANCE=4
else
  CORES_PER_INSTANCE=${CORES_PER_INSTANCE}
fi

source "$MODEL_DIR/models_v2/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
      --framework tensorflow \
      --precision ${PRECISION} \
      --mode ${MODE} \
      --model-name wide_deep \
      --batch-size ${BATCH_SIZE} \
      --data-location ${DATASET_DIR} \
      --output-dir ${OUTPUT_DIR} \
      --num-intra-threads=${CORES_PER_INSTANCE} \
      --num-inter-threads=1 \
      --numa-cores-per-instance=${CORES_PER_INSTANCE} \
      $@

if [[ $? == 0 ]]; then
  cat ${OUTPUT_DIR}/wide_deep_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | grep 'Throughput is:' | sed -e s"/.*: //"
  echo "Throughput summary:"
  grep 'Throughput is:' ${OUTPUT_DIR}/wide_deep_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | awk -F' ' '{sum+=$3;} END{print sum} '
  exit 0
else
  exit 1
fi
