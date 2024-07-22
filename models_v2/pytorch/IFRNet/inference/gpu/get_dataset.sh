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

if [ -d ${home_directory}/vimeo_interp_test ];
then
    echo "Vimeo-90K Triplet test dataset appears to be downloaded locally already"
    exit
fi

# --- Setup For Download ---
if [ ! -f "${home_directory}/vimeo_interp_test.zip" ]; then
    echo "Downloading Vimeo-90K Triplet test dataset."
    wget http://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip
else
  echo "Vimeo-90K Triplet test dataset is already downloaded."
fi

unzip ${home_directory}/vimeo_interp_test.zip

echo "Vimeo-90K Triplet test dataset is ready for use."

