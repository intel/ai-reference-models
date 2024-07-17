# Copyright (c) 2023-2024 Intel Corporation
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

#!/usr/bin/env bash

# setup.sh
#  - install OS pkgs
#  - should create virtual env & install pip  requirement.txt
#  - git clones & applying patches

set -e

mkdir -p build/result
cd build/result
if [[ ! -e fold_1.zip ]]; then
  wget https://zenodo.org/record/3904106/files/fold_1.zip
  unzip -o fold_1.zip
fi
cd ../..

mkdir -p folds
cd ./folds
for i in {0..4}
do
  wget https://raw.githubusercontent.com/mlcommons/inference/v3.1/vision/medical_imaging/3d-unet-brats19/folds/fold"$i"_validation.txt
done
cd ..
