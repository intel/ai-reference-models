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

# This script is to download required version of the Yolov5 model into yolov5

set -e

TIMEOUT=${TIMEOUT:-20}
YOLOV5_PUBLIC_REPO=${YOLOV5_PUBLIC_REPO:-https://github.com/ultralytics/yolov5}
YOLOV5_FULL_SHA=${YOLOV5_FULL_SHA:-781401ec70bc481b789b214003b722174e4b99e0}

YOLOV5_ARCHIVE_PATH=${YOLOV5_PUBLIC_REPO}/archive/${YOLOV5_FULL_SHA}.zip

# Fetch model source code
if [[ -d yolov5 ]]
then
    echo "The directory \"yolov5\" already exists. Skipping model download and performance patches for Intel XPUs"
    echo "To force the script to reinstall, remove the \"yolov5\" directory manually and rerun."
else
    wget -T ${TIMEOUT} -O yolov5.zip ${YOLOV5_ARCHIVE_PATH}
    unzip -o yolov5.zip
    mv yolov5-${YOLOV5_FULL_SHA} yolov5
    rm yolov5.zip
fi

