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
  echo "Please set PRECISION to fp32, int8 or bfloat16 or bfloat32."
  exit 1
fi

if [[ $PRECISION != "fp32" ]] && [[ $PRECISION != "int8" ]] && [[ $PRECISION != "bfloat16" ]] && [[ $PRECISION != "bfloat32" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32, bfloat16, bfloat32 and int8"
  exit 1
fi

if [ -z "${PRETRAINED_MODEL}" ]; then
  if [[ $PRECISION == "fp32" || $PRECISION == "bfloat32" ]]; then
    PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb"
  elif [[ $PRECISION == "int8" ]]; then
    PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/ssdmobilenet_int8_pretrained_model_combinedNMS_s8.pb"
  elif [[ $PRECISION == "bfloat16" ]]; then
    PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb"      
  else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, bfloat16, bfloat32 and int8"
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

MODE="inference"
CORES_PER_INSTANCE="socket"
if [ $PRECISION == "bfloat32" ]; then
  export ONEDNN_DEFAULT_FPMATH_MODE="BF16"
  PRECISION="fp32"
elif [ $PRECISION == "bfloat16" ]; then
  export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_ADD="BiasAdd,Relu6,Mul,AddV2"
  export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_REMOVE="BiasAdd,AddV2,Mul"
  export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_CLEARLIST_REMOVE="Relu6"
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="448"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name ssd-mobilenet \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --numa-cores-per-instance ${CORES_PER_INSTANCE} \
  --benchmark-only \
  $@

if [[ $? == 0 ]]; then
  echo "Summary total samples/sec:"
  grep 'Total samples/sec' ${OUTPUT_DIR}/ssd-mobilenet_${PRECISION}_inference_bs${BATCH_SIZE}_cores*_all_instances.log  | awk -F' ' '{sum+=$3;} END{print sum} '
else
  exit 1
fi

