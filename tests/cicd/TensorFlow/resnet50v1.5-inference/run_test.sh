#!/bin/bash
set -e
# Copyright (c) 2024 Intel Corporation
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
# ============================================================================

echo "Setup TensorFlow Test Enviroment for ResNet50 v1.5 Inference"

PRECISION=$1
SCRIPT=$2
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/TensorFlow/resnet50v1.5-inference/${SCRIPT}/${PRECISION}"}
WORKSPACE=$3
is_lkg_drop=$4
DATASET=$5

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${PRECISION}" == "bfloat16" ]]; then
  PRETRAINED_MODEL="/tf_dataset/pre-trained-models/resnet50v1_5/bf16/bf16_resnet50_v1.pb"
elif [[ "${PRECISION}" == "fp32" || "${PRECISION}" == "bfloat32" ]]; then
  PRETRAINED_MODEL="/tf_dataset/pre-trained-models/resnet50v1_5/fp32/resnet50_v1.pb"
elif [[ "${PRECISION}" == "int8" ]]; then
  PRETRAINED_MODEL="/tf_dataset/pre-trained-models/resnet50v1_5/int8/bias_resnet50.pb"
fi

if [[ "${is_lkg_drop}" == "true" ]]; then
  #export PATH=${WORKSPACE}/miniforge/bin:$PATH
  #source ${WORKSPACE}/tensorflow_setup/setvars.sh
  #source ${WORKSPACE}/tensorflow_setup/compiler/latest/env/vars.sh
  #source ${WORKSPACE}/tensorflow_setup/mkl/latest/env/vars.sh
  #source ${WORKSPACE}/tensorflow_setup/tbb/latest/env/vars.sh
  #source ${WORKSPACE}/tensorflow_setup/mpi/latest/env/vars.sh
  source ${WORKSPACE}/tensorflow_setup/bin/activate tensorflow
  #conda activate tensorflow
fi

# run following script
OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} PRETRAINED_MODEL=${PRETRAINED_MODEL} DATASET_DIR=${DATASET} ./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/${SCRIPT}
