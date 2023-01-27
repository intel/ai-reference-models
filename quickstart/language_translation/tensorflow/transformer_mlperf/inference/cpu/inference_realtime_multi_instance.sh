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

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to int8, fp32, bfloat32 or bfloat16."
  exit 1
elif [ ${PRECISION} != "int8" ] && [ ${PRECISION} != "fp32" ] && [ ${PRECISION} != "bfloat16" ] && [ ${PRECISION} != "bfloat32" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: int8, fp32, bfloat32 and bfloat16"
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

if [ -z "${PRETRAINED_MODEL}" ]; then
    if [[ $PRECISION == "fp32" ]] || [[ $PRECISION == "bfloat32" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/transformer_mlperf_fp32.pb"
    elif [[ $PRECISION == "bfloat16" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/transformer_mlperf_bf16.pb"
    elif [[ $PRECISION == "int8" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/transformer_mlperf_int8.pb"
    else
        echo "The specified precision '${PRECISION}' is unsupported."
        echo "Supported precisions are: int8, fp32, bfloat32 and bfloat16"
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

# Set up env variable for bfloat32
if [[ $PRECISION == "bfloat32" ]]; then
  ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
fi

MODE="inference"
CORES_PER_INSTANCE="4"

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

export TF_PATTERN_ALLOW_CTRL_DEPENDENCIES=1 

source "${MODEL_DIR}/quickstart/common/utils.sh"
_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name transformer_mlperf \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --numa-cores-per-instance ${CORES_PER_INSTANCE} \
  $@ \
  -- params=big file=uniformed_input_28tokens.en \
  file_out=translate_benchmark.txt \
  reference=uniformed_input_28tokens.de \
  vocab_file=vocab.ende.32768 \
  warmup_steps=3 steps=91

if [[ $? == 0 ]]; then
  cat ${OUTPUT_DIR}/transformer_mlperf_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | grep "Throughput:.*sentences/second" | sed -e s"/.*: //;s/sentences\/second//"
  echo "Throughput summary:"
  grep 'Throughput' ${OUTPUT_DIR}/transformer_mlperf_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | awk -F':' '{sum+=$2;} END{print sum} '
  exit 0
else
  exit 1
fi
