#!/bin/bash
# Copyright (c) 2022-2023 Intel Corporation
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

function Parser() {
    while [ $# -ne 0 ]; do
        case $1 in
            -m)
                shift
                MODEL="$1"
                ;;
            -d)
                shift
                DEVICE="$1"
                ;;
            -b)
                shift
                BATCH="$1"
                ;;
            -t)
                shift
                DTYPE="$1"
                ;;
            -n)
                shift
                if [ $1 -gt 0 ];then
                   NUM_ITER="$1"
                fi
                ;;
            -o)
                shift
                OUTPUT_DIR="$1"
                ;;
	    -s)
		shift
                dataset="$1"
                ;;
	    -w)
		shift
                model_path="$1"
                ;;
            -g)
                GDB_ARGS="gdb --args "
                ;;
	    -p)
		PROFILER_ARGS="--kineto_profile"
		;;
            --jit)
                JIT=true
                TRITON=false
                ;;
            --triton)
                TRITON=true
                JIT=false
                ;;
            -h | --help)
                echo "Usage: cmd_infer.sh [OPTION...] PAGE..."
                echo "-m, Optional    Specify the model type[bert_base or bert_large]. The default value is bert_base"
                echo "-d, Optional    Specify the device[cpu, xpu]. The default device is cpu"
                echo "-b, Optional    Specify the batch size. The default value is 32"
                echo "-t, Optional    Specify the dtype[FP32, FP16...]. The default value is FP32"
                echo "-n, Optional    Specify the number of iterations to run evaluation"
                echo "-o, Optional    Specify the output dir. The default value is /tmp/debug_squad/"
                echo "-g, Optional    use gdb"
		echo "-p, Optional    use PTI as profiler"
                echo "--triton, Optional use torch.compile to accelerate inference process (Conflict with --jit)"
                echo "--jit, Optional use jit to accelerate inference process (Conflict with --triton)"
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

MODEL="bert_base"
DEVICE=cpu
BATCH=32
DTYPE=FP32
NUM_ITER=-1
OUTPUT_DIR=/tmp/debug_squad/
GDB_ARGS=""
PROFILER_ARGS=""
NUMA_ARGS=""
TRITON=false
JIT=false
ACCELERATE_FLAG=""

Parser $@

if command -v numactl >& /dev/null ; then
if [ "x$MPI_LOCALRANKID" != "x" ] ; then
  REAL_NUM_NUMA_NODES=`lscpu | grep "NUMA node(s):" | awk '{print $NF}'`
  PPNUMA=$(( MPI_LOCALNRANKS / REAL_NUM_NUMA_NODES ))
  if [ $PPNUMA -eq 0 ] ; then 
    if [ "x$SINGLE_SOCKET_ONLY" == "x1" ] ; then 
      NUMA_ARGS="numactl -m 0 "
    fi
  else
    NUMARANK=$(( MPI_LOCALRANKID / PPNUMA ))
    NUMA_ARGS="$NUMA_ARGS $GDB_ARGS "
  fi
  NUM_RANKS=$PMI_SIZE
else
  NUMA_ARGS="numactl -m 0 "
  NUM_RANKS=1
fi
fi

# set dataset and model_path
if test -z $dataset || ! test -d $dataset ; then
  if test -d ./SQUAD1 ; then
    dataset=./SQUAD1
  else
    echo "Unable to find dataset path"
    exit 1
  fi
fi


if test -z $model_path || ! test -d $model_path ; then
  if [ "$MODEL" == "bert_base" ] ; then
    if test -d ./squad_base_finetuned_checkpoint ; then
      :
    else
      ./download_squad_base_fine_tuned_model.sh
    fi
    model_path=./squad_base_finetuned_checkpoint
  elif [ "$MODEL" == "bert_large" ] ; then
    if test -d ./squad_large_finetuned_checkpoint ; then
      :
    else
      ./download_squad_large_fine_tuned_model.sh
    fi
    model_path=./squad_large_finetuned_checkpoint
  else
    echo "The modle (${MODEL}) does not exist."
    exit
  fi
fi

if [ "$TRITON" == "true" ] ; then
  ACCELERATE_FLAG="--do_dynamo"
elif [ "$JIT" == "true" ] ; then
  ACCELERATE_FLAG="--do_jit"
fi

if [ "x$PROFILER_ARGS" != "x" ] ; then
  $NUMA_RAGS $GDB_ARGS python -u run_squad.py \
    --model_type bert \
    --model_name_or_path $model_path \
    --do_eval \
    --do_lower_case ${ACCELERATE_FLAG} \
    --device_choice ${DEVICE} \
    --dtype ${DTYPE}    \
    --predict_file $dataset/dev-v1.1.json \
    --per_gpu_eval_batch_size ${BATCH} \
    --max_seq_length 384 \
    --doc_stride 128 \
    --num_steps ${NUM_ITER} \
    --output_dir ${OUTPUT_DIR} \
    $PROFILER_ARGS
else
  $NUMA_RAGS $GDB_ARGS python -u run_squad.py \
    --model_type bert \
    --model_name_or_path $model_path \
    --do_eval \
    --do_lower_case ${ACCELERATE_FLAG} \
    --device_choice ${DEVICE} \
    --dtype ${DTYPE}    \
    --predict_file $dataset/dev-v1.1.json \
    --per_gpu_eval_batch_size ${BATCH} \
    --max_seq_length 384 \
    --doc_stride 128 \
    --num_steps ${NUM_ITER} \
    --output_dir ${OUTPUT_DIR}
fi
