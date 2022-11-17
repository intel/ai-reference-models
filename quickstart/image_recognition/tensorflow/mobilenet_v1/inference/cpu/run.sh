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

batch_size_env=""

if [ ! -z "${BATCH_SIZE}" ]; then
  batch_size_env="--env BATCH_SIZE=${BATCH_SIZE}"
fi

dataset_volume=""
dataset_env=""

if [ ! -z "${DATASET_DIR}" ]; then
  dataset_volume="--volume ${DATASET_DIR}:${DATASET_DIR}"
  dataset_env="--env DATASET_DIR=${DATASET_DIR}"
fi

IMAGE_NAME=${IMAGE_NAME:-model-zoo:tf-spr-mobilenet-v1-inference}
DOCKER_ARGS=${DOCKER_ARGS:---privileged --init -it}
WORKDIR=/workspace/tf-spr-mobilenet-v1-inference

# inference scripts:
# inference_realtime.sh
# inference_throughput.sh
# accuracy.sh
export SCRIPT="${SCRIPT:-inference_realtime.sh}"

if [[ ${SCRIPT} != quickstart* ]]; then
  SCRIPT="quickstart/$SCRIPT"
fi

docker run --rm \
  ${dataset_env} \
  ${batch_size_env} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  ${dataset_volume} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  -w ${WORKDIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
