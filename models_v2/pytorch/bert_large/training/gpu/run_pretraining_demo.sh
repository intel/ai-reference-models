#
# Copyright (c) 2022 Intel Corporation
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

python -u run_pretrain_mlperf.py \
    --config_name=bert_config.json \
    --input_dir=miniwiki/hdf5 \
    --output_dir=result \
    --eval_dir=miniwiki/hdf5 \
    --device=xpu \
    --do_train \
    --train_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --bf16 \
    --seed 123 \
    --sdp \
    --adamw --num-iterations 10

# DDP training
source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/setvars.sh
export LD_PRELOAD=$(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/lib/libmpi.so
export ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1

mpiexec -n 2 -l python -u run_pretrain_mlperf.py \
    --config_name=bert_config.json \
    --input_dir=miniwiki/hdf5 \
    --output_dir=result \
    --eval_dir=miniwiki/hdf5 \
    --device=xpu \
    --do_train \
    --train_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --bf16 \
    --seed 123 \
    --sdp \
    --adamw --num-iterations 10
