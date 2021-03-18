#!/bin/bash
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
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

if [ -z "${IMAGENET_REPO_PATH}" ]; then
  echo "The required environment variable IMAGENET_REPO_PATH has not been set"
  exit 1
fi

if [ ! -d "${IMAGENET_REPO_PATH}" ]; then
  echo "The IMAGENET_REPO_PATH '${IMAGENET_REPO_PATH}' does not exist"
  exit 1
fi

cd $IMAGENET_REPO_PATH
bash run_int8_multi_instance_ipex.sh resnet50 dnnl fp32 jit
