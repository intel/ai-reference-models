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


#!/bin/bash

bs=$1
datatype=$2

set -x
DATE=`date '+%Y%m%d'`

if [ ! -e logs/${DATE} ]; then
    mkdir -p logs/${DATE}
fi

if [ $ZE_TRACE_MODEL ]; then
    TRACE_MARCO="ze_tracer -c -h --chrome-device-stages --chrome-call-logging"
else
    TRACE_MARCO=""
fi

LOG_DIR="logs/${DATE}"
export PROFILE_PATH="${LOG_DIR}" # pass to script

# xpu
LOG_SUFFIX=("profile" "e2e")
IPEX_XPU_ONEDNN_LAYOUT=1 python -u unet_pp_xpu.py --batch_size ${bs} --dtype ${datatype} | tee ${LOG_DIR}/block.bs8.${LOG_SUFFIX}

# clean variable
unset PROFILE_PATH
