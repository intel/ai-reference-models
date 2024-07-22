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

#!/bin/bash

set -e

# --- Setup For Script Execution ---
home_directory=$(pwd)

# --- Setup For Script Execution ---
if [ ! -f "${home_directory}/ILSVRC2012_img_val.tar" ]; then
  echo "Downloading ImageNet validation set."
  wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
else
  echo "ImageNet validation set is already downloaded."
fi
if [ ! -f "${home_directory}/ILSVRC2012_devkit_t12.tar.gz" ]; then
  echo "Downloading ImageNet devkit set."
  wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
else
  echo "ImageNet devkit set is already downloaded."
fi
echo "ImageNet dataset is ready for use."

