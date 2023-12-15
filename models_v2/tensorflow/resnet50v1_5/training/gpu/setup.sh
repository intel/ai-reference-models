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

# setup.sh
#  - install OS pkgs
#  - should create virtual env & install pip  requirement.txt
#  - git clones & applying patches

set -e
apt-get update && apt-get install -y python3-venv protobuf-compiler
python3 -m venv $PWD/venv
. ./venv/bin/activate
pip install -r requirements.txt

#Prepare single tile model code
script_path="$(realpath "$0")"
workspace=$(dirname "$script_path")
mkdir $workspace/resnet50/ && cd $workspace/resnet50/
git clone -b v2.8.0 https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models
git apply $workspace/resnet50.patch
cd $workspace

#Prepare multi-tile model code 
mkdir $workspace/resnet50_hvd/ && cd $workspace/resnet50_hvd/
git clone -b v2.8.0 https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models
git apply $workspace/hvd_support.patch





