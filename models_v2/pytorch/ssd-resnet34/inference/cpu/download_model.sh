#!/usr/bin/env bash
### This file is originally from: [mlcommons repo](https://github.com/mlcommons/inference/tree/r0.5/others/cloud/single_stage_detector/download_model.sh)
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
dir=$(pwd)
mkdir -p ${CHECKPOINT_DIR}/pretrained; cd ${CHECKPOINT_DIR}/pretrained
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=13kWgEItsoxbVKUlkQz4ntjl1IZGk6_5Z'  -O 'resnet34-ssd1200.pth'
cd $dir
