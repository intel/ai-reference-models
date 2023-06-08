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

MODEL_DIR=${MODEL_DIR-$PWD}

if [ ! -e "${MODEL_DIR}/models/graph_classification/pytorch/inference/inference.py" ]; then
  echo "Could not find the script of inference.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the inference.py exist at the: \${MODEL_DIR}/models/graph_classification/pytorch/inference/inference.py"
  exit 1
fi


dir=$(pwd)
cd ${MODEL_DIR}/models/graph_classification/pytorch/inference/

pip uninstall pyg-lib -y && pip uninstall torch-scatter -y && pip uninstall torch-sparse -y && pip uninstall torch_geometric -y && pip uninstall ogb -y

# install pyg-lib
git clone https://github.com/pyg-team/pyg-lib.git && cd pyg-lib
git checkout master && git submodule sync && git submodule update --init --recursive && python setup.py install && cd ..
# install torch_geometric
git clone https://github.com/pyg-team/pytorch_geometric && cd pytorch_geometric
git checkout master && git submodule sync && git submodule update --init --recursive && pip install -e . && cd ..
# install ogb
git clone -b yanbing/products_profile https://github.com/yanbing-j/ogb.git && cd ogb && python setup.py install && cd ..
# install pytorch_scatter
git clone https://github.com/rusty1s/pytorch_scatter.git && cd pytorch_scatter
git checkout master && git submodule sync && git submodule update --init --recursive && python setup.py install && cd ..
# install pytorch_sparse
git clone https://github.com/rusty1s/pytorch_sparse.git && cd pytorch_sparse
git checkout master && git submodule sync && git submodule update --init --recursive && python setup.py install && cd ..

