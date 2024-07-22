#!/bin/bash

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

(( EUID != 0 )) && echo "fatal: must be executed under root" && exit 1

OUTPUT_DIR=${OUTPUT_DIR:-$PWD}

PROFILE=${PROFILE:-profiles/rife.fp16.csv}

PLATFORM=${PLATFORM:-Flex}
AMP=${AMP:-no}

if [[ ${PLATFORM} == "Flex" || ${PLATFORM} == "Max" ]]; then
  IMAGE=${IMAGE:-intel/image-interpolation:pytorch-flex-gpu-rife}
  platform_opt="--device /dev/dri/"
elif [[ ${PLATFORM} == "CUDA" ]]; then
  IMAGE=${IMAGE:-intel/image-interpolation:pytorch-cuda-gpu-rife}
  platform_opt="--gpus all"
else
  echo "fatal: unsupported platform: ${PLATFORM}"
  exit 1
fi

METADATA="$METADATA"
METADATA+=" config.system.docker0.docker.image=$IMAGE"
METADATA+=" config.system.docker0.docker.sha256=$(docker images --no-trunc --quiet $IMAGE | cut -d: -f2)"

# arguments in {} are mapped by benchmark.py from:
# 1. Profile given by --profile argument for each iteration of the test
# 2. Predefined variables: {output_dir}, {socket}
python3 -m benchmark --profile=$PROFILE --output_dir=$OUTPUT_DIR --telemetry --socket=/tmp/telemetry.s --platform=$PLATFORM --indent 4 --metadata "$METADATA" \
  docker run -it --rm --ipc=host --cap-add SYS_NICE $platform_opt \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/-e /') \
    -e PLATFORM=${PLATFORM} \
    -e AMP=${AMP} \
    -e NUM_INPUTS={num_inputs} \
    -e STREAMS={streams} \
    -e PRECISION={precision} \
    -e OUTPUT_DIR=/opt/output \
    -v {output_dir}:/opt/output \
    -e SOCKET=/tmp/telemetry.s \
    -v {socket}:/tmp/telemetry.s \
    $IMAGE \
      /bin/bash -c './run_model.sh --dummy'
