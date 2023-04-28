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

if [ -z "${PRETRAINED_MODEL}" ]; then
  echo "The required environment variable PRETRAINED_MODEL has not been set"
  exit 1
fi

if [ -z "${EVAL_DATA_FILE}" ]; then
  echo "The required environment variable EVAL_DATA_FILE has not been set"
  exit 1
fi

echo "PRETRAINED_MODEL: ${PRETRAINED_MODEL}"
IMAGE_NAME=${IMAGE_NAME:-model-zoo:pytorch-spr-bert-large-inference}
DOCKER_ARGS=${DOCKER_ARGS:---privileged --init -it}
WORKDIR=/workspace/pytorch-spr-bert-large-inference
EVAL_SCRIPT="${WORKDIR}/quickstart/transformers/examples/legacy/question-answering/run_squad.py"
mkdir -p $OUTPUT_DIR

# inference scripts:
# run_multi_instance_realtime.sh
# run_multi_instance_throughput.sh
# run_accuracy.sh
export SCRIPT="${SCRIPT:-run_multi_instance_realtime.sh}"

if [[ ${SCRIPT} != quickstart* ]]; then
  SCRIPT="quickstart/$SCRIPT"
fi

docker run --rm \
  ${dataset_env} \
  --env PRECISION=${PRECISION} \
  --env EVAL_DATA_FILE=${EVAL_DATA_FILE} \
  --env EVAL_SCRIPT=${EVAL_SCRIPT} \
  --env FINETUNED_MODEL=${PRETRAINED_MODEL} \
  --env INT8_CONFIG=${WORKDIR}/quickstart/configure.json \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${EVAL_DATA_FILE}:${EVAL_DATA_FILE} \
  --volume ${PRETRAINED_MODEL}:${PRETRAINED_MODEL} \
  --shm-size 8G \
  -w ${WORKDIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT ${PRECISION}
