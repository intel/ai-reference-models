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

echo 'MODEL_DIR='$MODEL_DIR
echo 'PRECISION='$PRECISION
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'PRETRAINED_DIR='$PRETRAINED_DIR
echo 'SQUAD_DIR='$SQUAD_DIR
echo 'FROZEN_GRAPH='$FROZEN_GRAPH

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[PRECISION]=${PRECISION}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[PRETRAINED_DIR]=${PRETRAINED_DIR}
input_envs[SQUAD_DIR]=${SQUAD_DIR}
input_envs[FROZEN_GRAPH]=${FROZEN_GRAPH}
input_envs[NUM_OAM]=${NUM_OAM}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

# Check for precision
if [[ ${PRECISION} == "fp16" || ${PRECISION} == "bfloat16" || ${PRECISION} == "fp32" ]]; then
  echo "The specified precision '${PRECISION}' is supported."
else
  echo "The specified precision '${PRECISION}' is not supported. Only fp16 and fp32 precision is supported"
  exit 1
fi

if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="64"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

if [[ ! -f "${FROZEN_GRAPH}" ]]; then
  frozen_graph=/workspace/tf-max-series-bert-large-inference/frozen_graph/fp32_bert_squad.pb
else
  frozen_graph=${FROZEN_GRAPH}
fi

if [[ ! -f "${frozen_graph}" ]]; then
  echo "Please set the path for frozen graph"
fi

if [[ $PRECISION == "fp16" ]]; then
  export ITEX_AUTO_MIXED_PRECISION=1
  export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="FLOAT16"
fi

if [[ $PRECISION == "bfloat16" ]]; then
  export ITEX_AUTO_MIXED_PRECISION=1
  export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="BFLOAT16"
fi

export CreateMultipleSubDevices=1
export TF_NUM_INTEROP_THREADS=1

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}
declare -a str

if [[ ${NUM_OAM} == "4" ]]; then
  echo "Bert-large Inference on ${NUM_OAM} OAM modules"
  NUM_TILES_PER_GPU=2
  cd ${MODEL_DIR}/models/language_modeling/tensorflow/bert_large/inference/gpu
  for i in $( eval echo {0..$((NUM_OAM-1))} )
    do
        for j in $( eval echo {0..$((NUM_TILES_PER_GPU-1))} )
            do
            str+=("ZE_AFFINITY_MASK="${i}"."${j}" python -u run_squad.py \
                  --vocab_file=${PRETRAINED_DIR}/vocab.txt \
                  --bert_config_file=${PRETRAINED_DIR}/bert_config.json \
                  --do_train=False \
                  --train_file=${SQUAD_DIR}/train-v1.1.json \
                  --do_predict=True \
                  --predict_file=${SQUAD_DIR}/dev-v1.1.json \
                  --predict_batch_size=${BATCH_SIZE} \
                  --learning_rate=3e-5 \
                  --num_train_epochs=2.0 \
                  --max_seq_length=384 \
                  --doc_stride=128 \
                  --precision=${PRECISION} \
                  --output_dir=${OUTPUT_DIR} \
                  --input_graph=${frozen_graph} \
                  --mode=${INFERENCE_MODE} 2>&1 | tee ${OUTPUT_DIR}/bertsquad_${PRECISION}_${INFERENCE_MODE}_c${i}_t${j}_raw.log & ")
            done
    done
  str=${str[@]}
  cmd_line=${str::-2}
  eval $cmd_line
else 
    echo "Currently only x4 OAM Modules are supported"
    exit 1
fi
