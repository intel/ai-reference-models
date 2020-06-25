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

# This script downloads and preprocesses the validation images for the COCO
# Dataset. It's intended to be used with the create_coco_tf_record.py script
# from the TensorFlow Model Garden, commit 1efe98bb8e8d98bbffc703a90d88df15fc2ce906.
# That particular commit is used, because later versions of the script create
# shards, which our models don't typically deal with.
#
# NOTE: This ONLY downloads and pre-processes validation images


if [ -z "${DATASET_DIR}" ]; then
  echo "ERROR: The DATASET_DIR var needs to be defined."
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  # Create the dataset directory, if it doesn't already exist
  mkdir ${DATASET_DIR}
fi

if [ -z "${TF_MODELS_DIR}" ]; then
  echo "ERROR: The TF_MODELS_DIR var needs to be defined to point to a clone of the tensorflow/models git repo"
  exit 1
fi

cd ${DATASET_DIR}

if [ ! -d "val2017" ]; then
  wget http://images.cocodataset.org/zips/val2017.zip
  unzip val2017.zip
  rm val2017.zip
fi

if [ ! -d "annotations" ]; then
  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P annotations
  unzip annotations/annotations_trainval2017.zip
  rm annotations/annotations_trainval2017.zip
fi

# empty dir and json for train/test image preprocessing
mkdir -p empty_dir
echo "{ \"images\": {}, \"categories\": {}}" > annotations/empty.json

cd ${TF_MODELS_DIR}/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd ${TF_MODELS_DIR}/research/object_detection/dataset_tools
python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${DATASET_DIR}/empty_dir" \
      --val_image_dir="${DATASET_DIR}/val2017" \
      --test_image_dir="${DATASET_DIR}/empty_dir" \
      --train_annotations_file="${DATASET_DIR}/annotations/empty.json" \
      --val_annotations_file="${DATASET_DIR}/annotations/instances_val2017.json" \
      --testdev_annotations_file="${DATASET_DIR}/annotations/empty.json" \
      --output_dir="${DATASET_DIR}/tf_records"

# remove directories that are no longer needed
rm -rf ${DATASET_DIR}/annotations
rm -rf ${DATASET_DIR}/empty_dir

# since we only grab the validation dataset, the TF records files for train
# and test images are size 0. Delete those to prevent confusion.
rm -f ${DATASET_DIR}/tf_records/coco_testdev.record
rm -f ${DATASET_DIR}/tf_records/coco_train.record
