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

echo 'MODEL_DIR='$MODEL_DIR
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'DATASET_DIR='$DATASET_DIR

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to int8, fp32, bfloat32, bfloat16 or fp16."
  exit 1
elif [ ${PRECISION} != "int8" ] && [ ${PRECISION} != "fp32" ] &&
     [ ${PRECISION} != "bfloat16" ] && [ ${PRECISION} != "fp16" ] &&
     [ ${PRECISION} != "bfloat32" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: int8, fp32, bfloat32, bfloat16 and fp16"
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
    elif [[ $PRECISION == "fp32" ]] || [[ $PRECISION == "bfloat32" ]] || [[ $PRECISION == "fp16" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/bert_large_fp32_pretrained_model.pb"
    else
        echo "The specified precision '${PRECISION}' is unsupported."
        echo "Supported precisions are: fp32, bfloat16, fp16, bfloat32 and int8"
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

source "${MODEL_DIR}/models_v2/common/utils.sh"
_get_numa_cores_lists
echo "Cores per node: ${cores_per_node}"

# If cores per instance env is not mentioned, then the workload will run with the default value.
if [ -z "${CORES_PER_INSTANCE}" ]; then
  CORES_PER_INSTANCE=${cores_per_node}
  echo "Runs an instance per ${CORES_PER_INSTANCE} cores."
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}" ]; then
  if [[ $PRECISION == "int8" ]]; then
    BATCH_SIZE="16"
  elif [[ $PRECISION == "bfloat16" ]] || [[ $PRECISION == "fp16" ]]; then
    BATCH_SIZE="32"
  elif [[ $PRECISION == "fp32" ]] || [[ $PRECISION == "bfloat32" ]]; then
    BATCH_SIZE="56"
  fi
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

# Set up env variable for bfloat32
if [[ $PRECISION == "bfloat32" ]]; then
  export ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
fi

if [ -z "${TF_THREAD_PINNING_MODE}" ]; then
  echo "TF_THREAD_PINNING_MODE is not set. Setting it to the following default value:"
  export TF_THREAD_PINNING_MODE=none,$(($CORES_PER_INSTANCE-1)),400
  echo "TF_THREAD_PINNING_MODE: $TF_THREAD_PINNING_MODE"
fi

if [ $PRECISION == "fp16" ]; then
  # Set environment variables needed to get best performance for fp16
  echo "Adding _FusedMatMul and _MklLayerNorm ops to AMP ALLOWLIST when running FP16."
  export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_ADD=_FusedMatMul,_MklLayerNorm
  echo "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_ADD=$TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_ADD"

fi

_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=bert_large \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework=tensorflow \
  --batch-size ${BATCH_SIZE} \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --checkpoint ${CHECKPOINT_DIR} \
  --accuracy-only \
  $@ \
  -- DEBIAN_FRONTEND=noninteractive \
  init_checkpoint=model.ckpt-3649 infer_option=SQuAD \
  experimental-gelu=True 2>&1 | tee ${OUTPUT_DIR}/bert_large_${PRECISION}_inference_bs${BATCH_SIZE}_accuracy.log

if [[ $? == 0 ]]; then
  echo "Accuracy:"
  cat ${OUTPUT_DIR}/bert_large_${PRECISION}_inference_bs${BATCH_SIZE}_accuracy.log | grep -ie "exact_match.*f1" | tail -n 1
  exit 0
else
  exit 1
fi

