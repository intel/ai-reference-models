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

pip install tensorboard
pip install datasets

# Check the operating system type
os_type=$(awk -F= '/^NAME/{print $2}' /etc/os-release)

# Install model specific dependencies:
if [[ "$os_name" == *"CentOS"* ]]; then
    yum install -y git-lfs
elif [[ "$os_name" == *"Ubuntu"* ]]; then
    apt install -y git-lfs
fi

cd quickstart/language_modeling/pytorch/bloom/inference/cpu
rm -rf transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.28.1
pip install -r requirements.txt
git apply ../../../../../../../models/language_modeling/pytorch/common/enable_ipex_for_transformers.diff
pip install -e ./
cd ..

git clone https://github.com/intel/torch-ccl.git
cd torch-ccl && git checkout v2.1.0+cpu
git submodule sync && git submodule update --init --recursive
pip install -e ./
cd ../

git clone https://github.com/delock/DeepSpeedSYCLSupport
cd DeepSpeedSYCLSupport
git checkout gma/run-opt-branch
python -m pip install -r requirements/requirements.txt
pip install -e ./
cd ../

git clone https://github.com/oneapi-src/oneCCL.git
cd oneCCL
mkdir build
cd build
cmake ..
make -j install
source _install/env/setvars.sh
cd ../..

export OUTPUT_TOKEN=32
export INPUT_TOKEN=32
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
