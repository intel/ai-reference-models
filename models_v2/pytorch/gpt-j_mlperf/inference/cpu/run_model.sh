#
# -*- coding: utf-8 -*-
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
#

#!/bin/bash

 
EXIST_MODEL=''
# Find model folders. Run the model with its folder in current path
model_list=("resnet50" "retinanet" "rnnt" "3d-unet" "bert" "gpt-j" "dlrm_2")
fd_list=("resnet50"  "retinanet"  "rnnt" "3d-unet-99.9"  "bert-99" "gptj-99" "dlrm-v2-99.9")
data_fd_list=("resnet50"  "retinanet/data"  "rnnt/mlperf-rnnt-librispeech" ""  "bert/dataset" "gpt-j/data" "dlrm_2/data_npy")
model_fd_list=("resnet50"  "retinanet/data"  "rnnt/mlperf-rnnt-librispeech" ""  "bert/model" "gpt-j/data" "dlrm_2/model")
dtype_list=(""  ""  "mix" ""  "" "int4" "")
impl_list=("" "" "" "" "" "" "pytorch-cpu-int8")
DTYPE=''
IMPL=''
for i in "${!fd_list[@]}"
do
   if [ -d ${fd_list[$i]} ]; then
       if [ -z "${MODEL_NAME}" ] || [ "${MODEL_NAME}" == "${model_list[$i]}" ] ; then
           echo ${fd_list[$i]} "Directory exists. index :" $i
           echo ${data_fd_list[$i]} " : DataPath index "
           echo ${model_fd_list[$i]} " : Model Path. index "
           EXIST_MODEL=${fd_list[$i]}

           if [ ! -z "${dtype_list[$i]}" ] ; then
               DTYPE=" -y ${dtype_list[$i]} "
           fi
           if [ ! -z "${impl_list[$i]}" ] ; then
               IMPL=" -i ${impl_list[$i]} "
           fi

           break
       fi
   fi
done

if [ -z "${EXIST_MODEL}" ]; then
    echo "Model Name is null."
    echo "export MODEL_NAME={resnet50,retinanet,rnnt,3d-unet,bert,gpt-j,dlrm_2,all}"
    exit 1
fi

if [ -z "${DATA_DIR}" ]; then
    echo "Path to dataset is null. Set the default dataset path as ./Dataset"
    DATA_DIR=~/Dataset
    echo "DATA_DIR : " $DATA_DIR
    
    if [ -d "${DATA_DIR}" ]; then
        mkdir -p ${DATA_DIR}
    fi
fi

if [ -z "${OUTPUT_DIR}" ]; then
    echo "Path to output is null. Set the default output path as ./Output"
    OUTPUT_DIR=~/Output
    echo "OUTPUT_DIR : " $OUTPUT_DIR
    
    if [ -d "${OUTPUT_DIR}" ]; then
        mkdir -p ${OUTPUT_DIR}
    fi
fi

if [ -z "${SUFFIX}" ]; then
    echo "SUFFIX is null. Set the default suffix as v0"
    SUFFIX=$(date +%Y-%m-%d)
    echo "export SUFFIX : " $SUFFIX
fi
cd $(pwd)/automation

python3 run.py -n ${fd_list[$i]} -d ${DATA_DIR}/${data_fd_list[$i]} -m ${DATA_DIR}/${model_fd_list[$i]} -t ${OUTPUT_DIR} -x ${SUFFIX} --performance-only --ci-run ${IMPL} ${DTYPE}

summary_log=${OUTPUT_DIR}/closed/Intel/results/*/${fd_list[$i]}/Offline/performance/run_1/mlperf_log_summary.txt
throughput=$(cat $summary_log | grep second: | sed -e "s/.*,//" | awk -F ' ' '{print $4}')

yaml_content=$(cat <<EOF
results:
 - key: throughput
   value: $throughput
   unit: it/s
EOF
)

# Write the content to a YAML file
echo "$yaml_content" >  ../results.yaml

summary_log=${OUTPUT_DIR}/closed/Intel/results/*/${fd_list[$i]}/Server/performance/run_1/mlperf_log_summary.txt
latency=$(cat $summary_log | grep second: | sed -e "s/.*,//" | awk -F ' ' '{print $6}')

yaml_content=$(cat <<EOF
results:
 - key: latency
   value: $latency
   unit: sec
EOF
)

# Write the content to a YAML file
echo "$yaml_content" >  ../latency_results.yaml
echo "YAML file created."
