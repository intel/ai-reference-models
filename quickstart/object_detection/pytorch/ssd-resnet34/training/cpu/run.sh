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

if [ -z "${BACKBONE_WEIGHTS}" ]; then
  echo "The required environment variable BACKBONE_WEIGHTS has not been set"
  exit 1
fi

IMAGE_NAME=${IMAGE_NAME:-model-zoo:pytorch-spr-ssd-resnet34-training}
DOCKER_ARGS=${DOCKER_ARGS:---privileged --init -it}
WORKDIR=/workspace/pytorch-spr-ssd-resnet34-training

# training scripts:
# train_performance.sh
# train_accuracy.sh
export SCRIPT="${SCRIPT:-train_performance.sh}"

if [[ ${SCRIPT} != quickstart* ]]; then
  SCRIPT="quickstart/$SCRIPT"
fi

docker run --rm \
  ${dataset_env} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env BACKBONE_WEIGHTS=${BACKBONE_WEIGHTS} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${BACKBONE_WEIGHTS}:${BACKBONE_WEIGHTS} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --shm-size 8G \
  -w ${WORKDIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
