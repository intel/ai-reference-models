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

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

echo "DATASET_DIR: ${DATASET_DIR}"
IMAGE_NAME=${IMAGE_NAME:-model-zoo:pytorch-spr-bert-large-training}
DOCKER_ARGS=${DOCKER_ARGS:---privileged --init -it}
WORKDIR=/workspace/pytorch-spr-bert-large-training
TRAIN_SCRIPT=${WORKDIR}/models/language_modeling/pytorch/bert_large/training/run_pretrain_mlperf.py

# training scripts:
# run_bert_pretrain_phase1.sh
# run_bert_pretrain_phase2.sh
export SCRIPT="${SCRIPT:-run_bert_pretrain_phase1.sh}"

if [[ ${SCRIPT} != quickstart* ]]; then
  SCRIPT="quickstart/$SCRIPT"
fi

config_file_env=""
config_file_mount=""
checkpoint_env=""
checkpoint_mount=""

if [ ! -z "${CONFIG_FILE}" ]; then
  echo "CONFIG_FILE: ${CONFIG_FILE}"
  config_file_env="--env BERT_MODEL_CONFIG=${CONFIG_FILE}"
  config_file_mount="--volume ${CONFIG_FILE}:${CONFIG_FILE}"
elif [[ ${SCRIPT} = *phase1* ]]; then
  echo "The required environment variable CONFIG_FILE has not been set to run phase1 pretraining."
  exit 1
fi

if [[ ${SCRIPT} = *phase2* ]]; then
  if [ ! -z "${CHECKPOINT_DIR}" ]; then
    echo "CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
    checkpoint_env="--env PRETRAINED_MODEL=${CHECKPOINT_DIR}"
    checkpoint_mount="--volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR}"
  else
    echo "The required environment variable CHECKPOINT_DIR has not been set to run phase2 pretraining."
    echo "Please set the CHECKPOINT_DIR var to point to the model_save directory from the phase 1 output directory."
    exit 1
  fi
fi

docker run --rm \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env TRAIN_SCRIPT=${TRAIN_SCRIPT} \
  --env DATASET_DIR=${DATASET_DIR} \
  ${config_file_env} \
  ${checkpoint_env} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  ${config_file_mount} \
  ${checkpoint_mount} \
  --shm-size 8G \
  -w ${WORKDIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT $PRECISION
