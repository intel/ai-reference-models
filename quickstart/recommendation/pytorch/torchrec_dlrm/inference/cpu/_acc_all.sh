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
export WEIGHT_DIR=/data/dlrm2_weight/model_weights
export DATASET_DIR=/data/dlrm_2_dataset/inference
export OUTPUT_DIR=./
export PLOTMEM=true

for precision in int8 fp32 fp16 bf32 bf16
do
export PRECISION=$precision
if [[ $PLOTMEM == "true" ]]; then
export MEMLOG=./log/${PRECISION}-acc-mem.dat
export MEMPIC=./log/${PRECISION}-acc-mem.jpeg
rm $MEMLOG
rm $MEMPIC
fi
bash test_accuracy.sh
done