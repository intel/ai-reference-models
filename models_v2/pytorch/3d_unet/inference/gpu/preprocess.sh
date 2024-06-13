# Copyright (c) 2023-2024 Intel Corporation
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

#!/usr/bin/env bash

if [[ -z "${DATASET_DIR}" ]]; then
  echo "Please set environment variable DATASET_DIR to <path/to/MICCAI_BraTS_2019_Data_Training> if you intend to use real dataset"
else
  DATASET_DIR="${DATASET_DIR}"
  RAW_DATA_DIR=$DATASET_DIR/"build/raw_data"
  PREPROCESSED_DATA_DIR=$DATASET_DIR/"build/preprocessed_data"
  POSTPROCESSED_DATA_DIR=$DATASET_DIR/"build/postprocessed_data"
  RESULT_DIR="build/result"
  export nnUNet_raw_data_base="$RAW_DATA_DIR"
  export nnUNet_preprocessed="$PREPROCESSED_DATA_DIR"
  export RESULTS_FOLDER="$RESULT_DIR"
  echo "Restructuring raw data to $RAW_DATA_DIR ..."
  mkdir -p "$RAW_DATA_DIR"
  python3 Task043_BraTS_2019.py --downloaded_data_dir "$DATASET_DIR" 
  echo "Preprocessing and saving preprocessed data to $PREPROCESSED_DATA_DIR ..."
  mkdir -p "$PREPROCESSED_DATA_DIR"
  python3 preprocess.py --model_dir "$RESULT_DIR"/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1 --raw_data_dir "$RAW_DATA_DIR"/nnUNet_raw_data/Task043_BraTS2019/imagesTr --preprocessed_data_dir "$PREPROCESSED_DATA_DIR"
  mkdir -p "$POSTPROCESSED_DATA_DIR"
fi
