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
if [ ! -f "${home_directory}/val2017.zip" ]; then
  echo "Downloading coco validation set."
  wget http://images.cocodataset.org/zips/val2017.zip
else
  echo "Coco validation set is already downloaded."
fi
echo "Coco dataset is ready for use."

