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

apt-get update
apt-get install -y --no-install-recommends --fix-missing git

git clone https://github.com/keras-team/keras-cv.git
cd keras-cv
git reset --hard 66fa74b6a2a0bb1e563ae8bce66496b118b95200
mv ../models/generative-ai/tensorflow/stable_diffusion/inference/gpu/patch .
git apply patch
cd - 
pip install matplotlib
pip install .

python -m pip install scikit-image
