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

export MODEL_DIR=${MODEL_DIR-$PWD}

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${DATA_PATH}" ]; then
  echo "The required environment variable DATA_PATH has not been set"
  exit 1
fi

if [ ! -d "${DATA_PATH}" ]; then
  echo "The DATA_PATH'${DATA_PATH}' does not exist"
  exit 1
fi

# TODO: Fill in the launch_benchmark.py command with the recommended args

${MODEL_DIR}/models/recommendation/pytorch/dlrm/training/bfloat16/bench/cleanup.sh
${MODEL_DIR}/models/recommendation/pytorch/dlrm/training/bfloat16/bench/dlrm_mlperf_4s_1n_cpx.sh

