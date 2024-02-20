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
weight=$2
bs=$3
datatype=$4

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
if [ $CUDA_HOME ]; then
    #cuda
    python -u main_no_ddp.py --eval --cfg configs/swin/swin_base_patch4_window7_224.yaml \
        --resume ${weight} --data-path ${DATASET} --local_rank 1 --batch-size ${bs} --num-iterations 20 --plain --device cuda 2>&1 | tee ${LOG_DIR}/block.bs${bs}.log
else
    # xpu
    PROFILE_ARGS=("--profile" " ")
    LOG_SUFFIX=("profile" "e2e")
    for i in $(seq ${#PROFILE_ARGS[@]}); do
        # IPEX_XPU_ONEDNN_LAYOUT=1 ${TRACE_MARCO} python -u main_no_ddp.py --eval --cfg configs/swin/swin_base_patch4_window7_224.yaml \
        #     --resume swin_base_patch4_window7_224.pth --data-path $DATASET --local_rank 1 --batch-size 1 --num-iterations 20 ${PROFILE_ARGS[i-1]} 2>&1 | tee ${LOG_DIR}/block.bs1.${LOG_SUFFIX[i-1]}.log

        # IPEX_XPU_ONEDNN_LAYOUT=1 ${TRACE_MARCO} python -u main_no_ddp.py --eval --cfg configs/swin/swin_base_patch4_window7_224.yaml \
        # --resume swin_base_patch4_window7_224.pth --data-path $DATASET --local_rank 1 --batch-size 8 --num-iterations 20 ${PROFILE_ARGS[i-1]} 2>&1 | tee ${LOG_DIR}/block.bs8.${LOG_SUFFIX[i-1]}.log

        IPEX_XPU_ONEDNN_LAYOUT=0 ${TRACE_MARCO} python -u main_no_ddp.py --eval --cfg configs/swin/swin_base_patch4_window7_224.yaml \
        --resume ${weight} --data-path ${DATASET} --local_rank 1 --batch-size ${bs} --dtype ${datatype} --num-iterations 20 ${PROFILE_ARGS[i-1]} --plain 2>&1 | tee ${LOG_DIR}/plain.bs${bs}.${LOG_SUFFIX[i-1]}.log
    done
fi

# clean variable
unset PROFILE_PATH

# For future ddp support. comment for now
# python -m torch.distributed.launch --nproc_per_node 1  main_xpu.py --eval \
# --cfg configs/swin/swin_base_patch4_window7_224.yaml --resume swin_base_patch4_window7_224.pth --data-path $DATASET

# mpirun -n 1 python main_xpu.py --eval  \
#  --cfg configs/swin/swin_base_patch4_window7_224.yaml --resume swin_base_patch4_window7_224.pth --data-path $DATASET
