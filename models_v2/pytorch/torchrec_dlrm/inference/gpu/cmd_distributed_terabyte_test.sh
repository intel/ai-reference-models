#!/bin/bash
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
set -x

function Parser() {
    while [ $# -ne 0 ]; do
        case $1 in
            -b)
                shift
                export GLOBAL_BATCH_SIZE="$1"
                ;;
            -fp16)
                shift
                FP16="$1"
                ;;
            -d)
                shift
                DATA="$1"
                ;;
            -m)
                shift
                WEIGHT="$1"
                ;;
            -nd)
                shift
                ND="$1"
                ;;
            -sp)
                shift
                SP="$1"
                ;;
            -tf32)
                shift
                TF32="$1"
                ;;
            -tv)
                shift
                TV="$1"
                ;;
            -h | --help)
                echo "Usage: cmd_infer.sh [OPTION...] PAGE..."
                echo "-b, Optional    Specify the batch size. The default value is 32768"
                echo "-fp16, Optional    Specify the input dtype is fp16. The default value is true"
                echo "-d, Optional    Specify the data file"
                echo "-m, Optional    Specify the weight file"
                echo "-nd, Optional    Specify the number of node"
                echo "-sp, Optional    Specify the sharding plan of embedding"
                echo "-tf32, Optional    Specify the input dtype is tf32. The default value is false"
                echo "-tv, Optional    Training with val. The default value is false"
                exit
                ;;
            --*|-*)
                echo ">>> New param: <$1>"
                ;;
            *)
                echo ">>> Parsing mismatch: $1"
                ;;
        esac
        shift
    done
}

torch_ccl_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $torch_ccl_path/env/setvars.sh
export MASTER_ADDR='127.0.0.1'
#export WORLD_SIZE=2 ;
export MASTER_PORT='10088'
export TOTAL_TRAINING_SAMPLES=4195197692;
export GLOBAL_BATCH_SIZE=65536;

ND=1
SP="round_robin"
#export CCL_LOG_LEVEL=DEBUG;
#export CCL_OP_SYNC=1

DATA=${DATA-'/home/sdp/xw/dlrm-v2/'}
WEIGHT=${WEIGHT-'/home/sdp/xw/model_weights'}

${FP16:=true}
${TF32:=false}
${TV:=false}
Parser $@
ARGS+=" --embedding_dim 128"
ARGS+=" --dense_arch_layer_sizes 512,256,128"
ARGS+=" --over_arch_layer_sizes 1024,1024,512,256,1"
ARGS+=" --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36"
ARGS+=" --validation_freq_within_epoch $((TOTAL_TRAINING_SAMPLES / (GLOBAL_BATCH_SIZE * 20 * 1000)))"
ARGS+=" --synthetic_multi_hot_criteo_path $DATA"
ARGS+=" --multi_hot_sizes 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1"
#ARGS+=" --multi_hot_distribution_type uniform"
ARGS+=" --use_xpu"
ARGS+=" --epochs 1"
ARGS+=" --pin_memory"
ARGS+=" --mmap_mode"
ARGS+=" --batch_size $GLOBAL_BATCH_SIZE"
ARGS+=" --interaction_type=dcn"
ARGS+=" --dcn_num_layers=3"
ARGS+=" --adagrad"
ARGS+=" --dcn_low_rank_dim=512"
ARGS+=" --numpy_rand_seed=12345"
ARGS+=" --log_freq 10"
ARGS+=" --amp"
ARGS+=" --inference_only"
ARGS+=" --snapshot_dir ${WEIGHT}"
ARGS+=" --limit_test_batches 50"
ARGS+=" --sharding_plan ${SP}"
ARGS+=" --num_nodes ${ND}"
ARGS+=" --learning_rate 0.005"

[ "$TV" = true ]             && ARGS+=" --train_with_val"
if [ "$TF32" = false ]; then
        [ "$FP16" = true ]             && ARGS+=" --fp16"
        echo "${ARGS}"
	mpirun -np 8 -ppn 8 --prepend-rank python -u dlrm_main.py  ${ARGS}
else
        echo "${ARGS}"
	IPEX_FP32_MATH_MODE=1 mpirun -np 8 -ppn 8 --prepend-rank python -u dlrm_main.py ${ARGS}
fi
