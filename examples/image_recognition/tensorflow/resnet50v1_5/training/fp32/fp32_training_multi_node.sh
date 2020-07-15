#!/usr/bin/env bash
#
# Copyright (c) 2020 Intel Corporation
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

export PYTHONPATH=$MODEL_DIR/models/image_recognition/tensorflow/resnet50v1_5/training:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/mpirun:$LD_LIBRARY_PATH

echo 'MODEL_DIR='$MODEL_DIR
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'DATASET_DIR='$DATASET_DIR
echo 'PYTHONPATH='$PYTHONPATH
echo 'LD_LIBRARY_PATH='$LD_LIBRARY_PATH

_allow_run_as_root=''
if (( $(id -u) == 0 )); then
  _allow_run_as_root='--allow-run-as-root'
fi

if [[ -z $MODEL_DIR ]]; then
  echo "The required environment variable MODEL_DIR has not been set" >&2
  exit 1
fi

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set" >&2
  exit 1
fi

if [[ ! -d $OUTPUT_DIR ]]; then
  # Create the output directory, if it doesn't already exist
  mkdir -p $OUTPUT_DIR
fi

if [[ -z $DATASET_DIR ]]; then
  echo "The required environment variable DATASET_DIR has not been set" >&2
  exit 1
fi

if [[ ! -d $DATASET_DIR ]]; then
  echo 'The DATASET_DIR '$DATASET_DIR' does not exist' >&2
  exit 1
fi

mpirun -x PYTHONPATH -x LD_LIBRARY_PATH $_allow_run_as_root -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -np "2" -bind-to none -map-by slot \
  python3 $MODEL_DIR/models/image_recognition/tensorflow/resnet50v1_5/training/mlperf_resnet/imagenet_main.py 2 \
    --batch_size 128 \
    --max_train_steps 5 \
    --train_epochs 1 \
    --epochs_between_evals 1 \
    --inter_op_parallelism_threads 2 \
    --intra_op_parallelism_threads 22 \
    --version 1 \
    --resnet_size 50 \
    --output_dir=$OUTPUT_DIR \
    --data_dir=$DATASET_DIR
