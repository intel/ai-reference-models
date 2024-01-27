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

#converge
export Train_DATASET=/pytorch/bert_dataset_from_mlperf_scp/hdf5/training-4320/hdf5_4320_shards_varlength
export Eval_DATASET=/pytorch/bert_dataset_from_mlperf_scp/hdf5/eval_varlength
export Output_RESULT=./result
# ckpt trained on phase1, load it to run pretrain phase2
export MODEL_NAME_PATH=/pytorch/bert_dataset_from_mlperf_scp/checkpoint
export CKPT_PATH=./result

echo "Bert single tile convergence"
echo "[info] Train_DATASET:"
printenv Train_DATASET
echo "[info] Eval_DATASET:"
printenv Eval_DATASET
echo "[info] Output_RESULT:"
printenv Output_RESULT
echo "[info] MODEL_NAME_PATH:"
printenv MODEL_NAME_PATH
echo "[info] CKPT_PATH:"
printenv CKPT_PATH

# max_steps means how many steps this training will run
# max_steps_for_scheduler means how many steps the scheduler will use to decrease the lr
# min_learning_rate means the minial lr the scheduler will decrease the lr to
# learning_rate means the initial lr

# [watch out] min learning rate means the minimal learning rate the scheduler will decay lr to
python -u run_pretrain_mlperf.py \
    --input_dir=$Train_DATASET \
    --output_dir=$Output_RESULT \
    --eval_dir=$Eval_DATASET \
    --model_name_or_path=$MODEL_NAME_PATH \
    --device=xpu \
    --do_train \
    --train_batch_size=16 \
    --eval_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --bf16 \
    --lamb \
    --phase2 \
    --workers 4 \
    --amp \
    --converge \
    --opt_lamb_beta_1=0.9 \
    --opt_lamb_beta_2=0.999 \
    --warmup_steps=0 \
    --start_warmup_step=0 \
    --learning_rate=2e-4 \
    --min_learning_rate=9e-6 \
    --weight_decay_rate=0.01 \
    --max_steps=1313280 \
    --max_steps_for_scheduler=1094400 \
    --max_samples_termination=21012480 \
    --eval_iter_start_samples=50000 \
    --eval_iter_samples=50000 \
    --num_samples_per_checkpoint 50000 \
    --min_samples_to_start_checkpoints 10000 \
    --keep_n_most_recent_checkpoints 10 \
    --seed 123
    # --resume_from_checkpoint \
    # --resume_checkpoint $CKPT_PATH
