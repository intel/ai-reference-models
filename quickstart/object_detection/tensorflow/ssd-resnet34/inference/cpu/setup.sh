#!/usr/bin/env bash
#
# Copyright (c) 2022 Intel Corporation
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

# Install libgl1-mesa-glx, libglib2.0-0, python3-dev
yum update -y && yum install -y mesa-libGL glib2-devel  python-devel

# Install model dependencies
pip install -r ${MODEL_DIR}/benchmarks/object_detection/tensorflow/ssd-resnet34/requirements.txt

