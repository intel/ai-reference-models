#!/usr/bin/env bash
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

export MODEL_DIR=${MODEL_DIR-$PWD}

HOROVOD_PATCH=${MODEL_DIR}/quickstart/language_modeling/tensorflow/bert_large/training/gpu/hvs_support.patch

cd ${MODEL_DIR}
git apply ${HOROVOD_PATCH}
cd -
