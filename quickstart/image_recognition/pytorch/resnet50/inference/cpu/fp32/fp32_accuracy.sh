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

source "${MODEL_DIR}/quickstart/common/utils.sh"
_get_platform_type
TCMALLOC_ARGS=""
ARGS=""

if [[ ${PLATFORM} == "linux" ]]; then
  pip list | grep intel-extension-for-pytorch
  if [[ "$?" == 0 ]]; then
    TCMALLOC_ARGS=" -m intel_extension_for_pytorch.cpu.launch --enable_tcmalloc"
    # in case IPEX is used we set ipex and jit path args
    ARGS="--ipex --jit"
    echo "Running using ${ARGS} args ..."
  fi
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

python ${TCMALLOC_ARGS} \
    models/image_recognition/pytorch/common/main.py \
    --arch resnet50 ${DATASET_DIR} \
    --evaluate \
    --pretrained \
    ${ARGS} \
    --workers 0 \
    --batch-size 128
