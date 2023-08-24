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

export MODEL_DIR=/home/haozhe/lz/frameworks.ai.models.intel-models
export DATASET_DIR=/data/dlrm_2_dataset/inference
export OUTPUT_DIR=./
export ENABLE_TORCH_PROFILE=true
export PLOTMEM=true

seq=`date +%m%d%H%M%S`
mkdir log/${seq}/
for precision in int8 fp32 fp16 bf32 bf16
do
export PRECISION=$precision
if [[ $PLOTMEM == "true" ]]; then
export MEMLOG=./log/${seq}/${PRECISION}-bench-real-mem.dat
export MEMPIC=./log/${seq}/${PRECISION}-bench-real.jpeg
fi
bash inference_performance.sh
done


unset DATASET_DIR
for precision in int8 fp32 fp16 bf32 bf16
do
export PRECISION=$precision
if [[ $PLOTMEM == "true" ]]; then
export MEMLOG=./log/${seq}/${PRECISION}-bench-dummy-mem.dat
export MEMPIC=./log/${seq}/${PRECISION}-bench-dummy-mem.jpeg
fi
bash inference_performance.sh 2>&1 |tee ./log/${PRECISION}-bench-dummy.log
done