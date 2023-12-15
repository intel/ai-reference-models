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

script_path="$(realpath "$0")"
workspace=$(dirname "$script_path")
#Prepare single tile model code
mkdir $workspace/3d_unet/ && cd $workspace/3d_unet/
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples
git checkout 88eb3cff2f03dad85035621d041e23a14345999e
cd TensorFlow/Segmentation/UNet_3D_Medical/
git apply $workspace/3dunet_itex.patch
cd $workspace

#Prepare multi-tile model code 
mkdir $workspace/3d_unet_hvd/ && cd $workspace/3d_unet_hvd/
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples
git checkout 88eb3cff2f03dad85035621d041e23a14345999e
cd TensorFlow/Segmentation/UNet_3D_Medical/
git apply $workspace/3dunet_itex_with_horovod.patch





