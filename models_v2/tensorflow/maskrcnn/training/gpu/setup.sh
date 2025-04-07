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

# Download and run the Poetry installation script
../../../../common/install-python-dependencies.sh

current_dir=$(pwd)
if [ -d "DeepLearningExamples" ]; then
  echo "Repository already exists. Skipping clone."
else
  git clone https://github.com/NVIDIA/DeepLearningExamples.git
  cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN
  git checkout 9dd9fcb98f56187e49c5ee280cf8dbd530dde57b
  git apply  $current_dir/EnableBF16.patch
  cd -
fi
