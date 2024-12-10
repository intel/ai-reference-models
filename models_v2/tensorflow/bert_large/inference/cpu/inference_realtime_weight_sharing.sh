#!/usr/bin/env bash
#
# Copyright (c) 2022 Intel Corporation
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

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  # Unzip the squad checkpoint files
  pretrained_model_dir="pretrained_model/bert_large_checkpoints"
  if [ ! -d "${pretrained_model_dir}" ]; then
    unzip pretrained_model/bert_large_checkpoints.zip -d pretrained_model
  fi
  CHECKPOINT_DIR="${MODEL_DIR}/${pretrained_model_dir}"
fi

# Create an array of input directories that are expected and then verify that they exist
declare -A input_dirs
input_dirs[CHECKPOINT_DIR]=${CHECKPOINT_DIR}
input_dirs[DATASET_DIR]=${DATASET_DIR}

for i in "${!input_dirs[@]}"; do
  var_name=$i
  dir_path=${input_dirs[$i]}

  if [[ -z $dir_path ]]; then
    echo "The required environment variable $var_name is empty" >&2
    exit 1
  fi

  if [[ ! -d $dir_path ]]; then
    echo "The $var_name path '$dir_path' does not exist" >&2
    exit 1
  fi
done
if [ -z "${PRETRAINED_MODEL}" ]; then
    if [[ $PRECISION == "int8" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/bert_large_int8_pretrained_model.pb"
    elif [[ $PRECISION == "bfloat16" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/bert_large_bfloat16_pretrained_model.pb"
    elif [[ $PRECISION == "fp32" ]] || [[ $PRECISION == "bfloat32" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/bert_large_fp32_pretrained_model.pb"
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

# If cores per instance env is not mentioned, then the workload will run with the default value.
if [ -z "${CORES_PER_INSTANCE}" ]; then
  CORES_PER_INSTANCE="4"
  echo "Running with default ${CORES_PER_INSTANCE} cores per instance"
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

# Set up env variable for bfloat32
if [[ $PRECISION == "bfloat32" ]]; then
  export ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
fi

# If OMP_NUM_THREADS env is not mentioned, then run with the default value
if [ -z "${OMP_NUM_THREADS}" ]; then
  export OMP_NUM_THREADS=4
else
  export OMP_NUM_THREADS=${OMP_NUM_THREADS}
fi

# clean up old log files if found
rm -rf ${OUTPUT_DIR}/Bert_large_${PRECISION}_bs${BATCH_SIZE}_Latency_inference_instance_*

source "${MODEL_DIR}/models_v2/common/utils.sh"
_ht_status_spr
_get_socket_cores_lists
_command numactl --localalloc --physcpubind=${cores_per_socket_arr[0]} python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=bert_large \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --checkpoint ${CHECKPOINT_DIR} \
  --num-intra-threads ${cores_per_socket} \
  --num-inter-threads -1 \
  --weight-sharing \
  --warmpup-steps=100 \
  --steps=200 \
  --benchmark-only \
  --verbose \
  $@ \
  infer-option=SQuAD >> ${OUTPUT_DIR}/Bert_large_${PRECISION}_bs${BATCH_SIZE}_Latency_inference_instance_0.log 2>&1 & \
numactl --localalloc --physcpubind=${cores_per_socket_arr[1]} python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=bert_large \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --checkpoint ${CHECKPOINT_DIR} \
  --num-intra-threads ${cores_per_socket} \
  --num-inter-threads -1 \
  --weight-sharing \
  --warmpup-steps=100 \
  --steps=200 \
  --benchmark-only \
  --verbose \
  $@ \
  infer-option=SQuAD >> ${OUTPUT_DIR}/Bert_large_${PRECISION}_bs${BATCH_SIZE}_Latency_inference_instance_1.log 2>&1 & \
  wait

if [[ $? == 0 ]]; then
  cat ${OUTPUT_DIR}/Bert_large_${PRECISION}_bs${BATCH_SIZE}_Latency_inference_instance_*.log | grep -i "Total throughput" | sed -e s"/Total //"
  echo "Total Throughput:"
  grep -i 'Total Throughput' ${OUTPUT_DIR}/Bert_large_${PRECISION}_bs${BATCH_SIZE}_Latency_inference_instance_*.log | awk -F': ' '{sum+=$2;} END{print sum} '
  exit 0
else
  exit 1
fi
