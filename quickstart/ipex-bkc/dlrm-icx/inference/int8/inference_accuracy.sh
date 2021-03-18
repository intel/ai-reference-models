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

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${WEIGHT_PATH}" ]; then
  echo "The required environment variable WEIGHT_PATH has not been set"
  exit 1
fi

if [ ! -f "${WEIGHT_PATH}" ]; then
  echo "The WEIGHT_PATH '${WEIGHT_PATH}' does not exist"
  exit 1
fi

export DATASET_PATH=$DATASET_DIR
cd quickstart/dlrm
bash run_inference_accuracy.sh int8
