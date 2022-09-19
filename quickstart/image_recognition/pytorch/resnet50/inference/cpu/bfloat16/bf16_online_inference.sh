#!/usr/bin/env bash
#
# Copyright (c) 2020 Intel Corporation
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

source "${MODEL_DIR}/quickstart/common/utils.sh"
_get_platform_type
MULTI_INSTANCE_ARGS=""
ARGS=""

if [[ ${PLATFORM} == "linux" ]] && [[ ! -z "${ARGS}" ]]; then
    pip list | grep intel-extension-for-pytorch
    if [[ "$?" == 0 ]]; then
        CORES=`lscpu | grep Core | awk '{print $4}'`
        SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
        TOTAL_CORES=`expr $CORES \* $SOCKETS`
        CORES_PER_INSTANCE=4
        INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
        MULTI_INSTANCE_ARGS=" -m intel_extension_for_pytorch.cpu.launch --enable_tcmalloc --ninstances ${INSTANCES} --ncore_per_instance ${CORES_PER_INSTANCE}"

        echo "Running '${INSTANCES}' instance"
        # in case IPEX is used we set ipex and jit path args
        ARGS="--ipex --jit"
        echo "Running using ${ARGS} args ..."
    fi
fi

python ${MULTI_INSTANCE_ARGS} \
    models/image_recognition/pytorch/common/main.py \
    --arch resnet50 ../ \
    --evaluate \
    ${ARGS} \
    --bf16 \
    --batch-size 1 \
    --dummy
