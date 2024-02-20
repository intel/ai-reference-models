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
# SPDX-License-Identifier: EPL-2.0
#

#!/bin/bash

DATASET=$1
BS=$2
DATATYPE=$3

set -x
DATE=`date '+%Y%m%d'`

#if [ ! -e logs/${DATE} ]; then
#    mkdir -p logs/${DATE}
#fi

#LOG_DIR="logs/${DATE}"
if [ $CUDA_HOME ]; then
    #cuda
    python -u main_no_ddp.py --cfg configs/swin/swin_base_patch4_window7_224.yaml \
     --data-path ${DATASET} --local_rank 1 --batch-size ${BS} --dtype ${DATATYPE} --plain --device cuda
else
    #xpu
    if [ ${DATATYPE} == "tf32" ]; then
        IPEX_FP32_MATH_MODE=1 python -u main_no_ddp.py --cfg configs/swin/swin_base_patch4_window7_224.yaml \
         --data-path ${DATASET} --local_rank 1 --batch-size ${BS} --dtype ${DATATYPE} --max_epochs 1 --device xpu --plain
    #fp32&bf16
    else
        python -u main_no_ddp.py --cfg configs/swin/swin_base_patch4_window7_224.yaml \
         --data-path ${DATASET} --local_rank 1 --batch-size ${BS} --dtype ${DATATYPE} --max_epochs 1 --device xpu --plain
    fi
fi
