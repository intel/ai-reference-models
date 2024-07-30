#!/usr/bin/env bash
#
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
#

MODEL_DIR=${MODEL_DIR-$PWD}

if [ ! -e "${MODEL_DIR}/inference.py" ]; then
  echo "Could not find the script of inference.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the inference.py exist at the: \${MODEL_DIR}/inference.py"
  exit 1
fi

dir=$(pwd)
pip install -r requirements.txt
pip install unidecode inflect
pip install librosa==0.9.1

# warp-transducer:
git clone https://github.com/HawkAaron/warp-transducer
cd warp-transducer
git checkout master
git apply ${MODEL_DIR}/enable_warprnnt_c++17.diff
rm -rf build
mkdir build; cd build
cmake .. 
make 
cd ../pytorch_binding
pip install -e .
cd $dir
