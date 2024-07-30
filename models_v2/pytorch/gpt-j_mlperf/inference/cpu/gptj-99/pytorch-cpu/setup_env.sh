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
CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data

export CALIBRATION_DATA_JSON=${WORKLOAD_DATA}/calibration-data/cnn_dailymail_calibration.json

export CHECKPOINT_DIR=${WORKLOAD_DATA}/gpt-j-checkpoint

export VALIDATION_DATA_JSON=${WORKLOAD_DATA}/validation-data/cnn_dailymail_validation.json

export INT8_MODEL_DIR=${WORKLOAD_DATA}/gpt-j-int8-model

export INT4_MODEL_DIR=${WORKLOAD_DATA}/gpt-j-int4-model

export INT4_CALIBRATION_DIR=${WORKLOAD_DATA}/quantized-int4-model

mkdir -p ${INT8_MODEL_DIR}
mkdir -p ${INT4_MODEL_DIR}
