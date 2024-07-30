#
# -*- coding: utf-8 -*-
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
#

#!/bin/bash

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[DATA_DIR]=${DATA_DIR}
input_envs[RESULTS_DIR]=${RESULTS_DIR}
input_envs[DATATYPE]=${DATATYPE}
input_envs[MULTI_TILE]=${MULTI_TILE}
input_envs[NUM_DEVICES]=${NUM_DEVICES}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}

  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

if [ -d ${DATA_DIR} ];then
  echo "DATA_DIR is: "${DATA_DIR}
else
  echo "Error: the path of dataset does not exist!"
  exit 1
fi

if [ ${DATATYPE} == "tf32" ];then
  export ITEX_FP32_MATH_MODE=TF32
fi


echo 'Running with parameters:'
echo " DATA_DIR: ${DATA_DIR}"
echo " RESULTS_DIR: ${RESULTS_DIR}"
echo " DATATYPE: ${DATATYPE}"
echo " MULTI_TILE: $MULTI_TILE"
echo " NUM_DEVICES: $NUM_DEVICES"


rm -fr $RESULTS_DIR

pwd=$PWD
cd ./DeepLearningExamples/TensorFlow2/LanguageModeling/BERT

export ITEX_OPS_OVERRIDE=1
export DATA_DIR=$DATA_DIR
if [ "$DATATYPE" == "bf16" ]; then
    TRAIN_BATCH_SIZE_PHASE1=60
    TRAIN_BATCH_SIZE_PHASE2=32
elif [ "$DATATYPE" == "fp32" ]; then
    TRAIN_BATCH_SIZE_PHASE1=30
    TRAIN_BATCH_SIZE_PHASE2=16
elif [ "$DATATYPE" == "tf32" ]; then
    TRAIN_BATCH_SIZE_PHASE1=30
    TRAIN_BATCH_SIZE_PHASE2=16
else  
    echo "not support datatype"
fi
EVAL_BATCH_SIZE=8
LEARNING_RATE_PHASE1=7.5e-4
LEARNING_RATE_PHASE2=5e-4
DATATYPE=$DATATYPE
USE_XLA=false
WARMUP_STEPS_PHASE1=5
WARMUP_STEPS_PHASE2=1
TRAIN_STEPS=20
SAVE_CHECKPOINT_STEPS=2
NUM_ACCUMULATION_STEPS_PHASE1=64
NUM_ACCUMULATION_STEPS_PHASE2=30
BERT_MODEL=large


bash scripts/run_pretraining_lamb_phase2.sh \
    $TRAIN_BATCH_SIZE_PHASE1 \
    $TRAIN_BATCH_SIZE_PHASE2 \
    $EVAL_BATCH_SIZE \
    $LEARNING_RATE_PHASE1 \
    $LEARNING_RATE_PHASE2 \
    $DATATYPE \
    $USE_XLA \
    $NUM_DEVICES \
    $WARMUP_STEPS_PHASE1 \
    $WARMUP_STEPS_PHASE2 \
    $TRAIN_STEPS \
    $SAVE_CHECKPOINT_STEPS \
    $NUM_ACCUMULATION_STEPS_PHASE1 \
    $NUM_ACCUMULATION_STEPS_PHASE2 \
    $BERT_MODEL \
    $DATA_DIR \
    $RESULTS_DIR \
    |& tee $RESULTS_DIR/bert_large_training_${DATATYPE}.log

cd -

throughput=$(cat $RESULTS_DIR/bert_large_training_${DATATYPE}.log | grep "Throughput Average (sequences/sec)"  | tail -n 1 | awk -F ' ' '{print $9}')

yaml_content=$(cat <<EOF
results:
 - key: throughput
   value: $throughput
   unit: images/sec
EOF
)

# Write the content to a YAML file
echo "$yaml_content" >  ${RESULTS_DIR}/results.yaml
echo "YAML file created."
