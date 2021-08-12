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

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, int8, or bf16."
  exit 1
fi

cd ${MODEL_DIR}/models/resnet50/examples/imagenet

# download pretrained weight.
python hub_help.py --url https://download.pytorch.org/models/resnet50-0676ba61.pth

export work_space=${OUTPUT_DIR}

if [[ $PRECISION == "int8" ]]; then
    bash run_multi_instance_ipex_spr.sh resnet50 int8 no_jit resnet50_configure.json 2>&1 | tee -a ${OUTPUT_DIR}/resnet50-inference-throughput-int8.log
elif [[ $PRECISION == "bf16" ]]; then
    bash run_multi_instance_ipex_spr.sh resnet50 bf16 jit 2>&1 | tee -a ${OUTPUT_DIR}/resnet50-inference-throughput-bf16.log
elif [[ $PRECISION == "fp32" ]]; then
    bash run_multi_instance_ipex_spr.sh resnet50 fp32 jit 2>&1 | tee -a ${OUTPUT_DIR}/resnet50-inference-throughput-fp32.log
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, bf16, and int8"
    exit 1
fi