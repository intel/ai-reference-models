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

# This script preprocesses the training images for the COCO Dataset to create
# TF records files. The raw training images and annotations must be downloaded
# prior to running this script (https://cocodataset.org/#download).
#
# The following vars need to be set:
#   IMAGE_DIR: Points to the raw training images (extracted from train2017.zip)
#   ANNOTATIONS_DIR: Points to the annotations (extracted from annotations_trainval2017.zip)
#
# This is intended to be used with the create_coco_tf_record.py script from the
# TensorFlow Model Garden.
#
# NOTE: This pre-processes the training images only

# If the DATASET_DIR is set, then ensure it exists and set paths for the images and annotations
if [[ ! -z "${DATASET_DIR}" ]]; then
  if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "ERROR: The specified DATASET_DIR ($DATASET_DIR) does not exist."
    exit 1
  fi

  IMAGE_DIR=${DATASET_DIR}/train2017
  ANNOTATIONS_DIR=${DATASET_DIR}/annotations
fi

# Verify that the a directory exists for the raw train images
if [[ ! -d "${IMAGE_DIR}" ]]; then
  echo "ERROR: The IMAGE_DIR (${IMAGE_DIR}) does not exist. This var needs to point to the raw coco train images."
  exit 1
fi

# Verify that the a directory exists for the annotations
if [[ ! -d "${ANNOTATIONS_DIR}" ]]; then
  echo "ERROR: The ANNOTATIONS_DIR (${ANNOTATIONS_DIR}) does not exist. This var needs to point to the coco annotations directory."
  exit 1
fi

# Verify that we have the path to the tensorflow/models code
if [[ ! -d "${TF_MODELS_DIR}" ]]; then
  echo "ERROR: The TF_MODELS_DIR var needs to be defined to point to a clone of the tensorflow/models git repo"
  exit 1
fi

# Checkout the specified branch for the tensorflow/models code
if [[ ! -z "${TF_MODELS_BRANCH}" ]]; then
  cd ${TF_MODELS_DIR}
  git checkout ${TF_MODELS_BRANCH}
fi

# Set the PYTHONPATH
cd ${TF_MODELS_DIR}/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Create empty dir and json for val/test image preprocessing, so that we don't require
# the user to also download val/test images when all that's needed is training images.
EMPTY_DIR=${DATASET_DIR}/empty_dir
EMPTY_ANNOTATIONS=${DATASET_DIR}/empty.json
mkdir -p ${EMPTY_DIR}
echo "{ \"images\": {}, \"categories\": {}}" > ${EMPTY_ANNOTATIONS}

cd ${TF_MODELS_DIR}/research/object_detection/dataset_tools
python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${IMAGE_DIR}" \
      --val_image_dir="${EMPTY_DIR}" \
      --test_image_dir="${EMPTY_DIR}" \
      --train_annotations_file="${ANNOTATIONS_DIR}/instances_train2017.json" \
      --val_annotations_file="${EMPTY_ANNOTATIONS}" \
      --testdev_annotations_file="${EMPTY_ANNOTATIONS}" \
      --output_dir="${DATASET_DIR}"

# remove dummy directory and annotations file
rm -rf ${EMPTY_DIR}
rm -rf ${EMPTY_ANNOTATIONS}

# since we only grab the train dataset, the TF records files for validation
# and test images are size 0. Delete those to prevent confusion.
rm -f ${DATASET_DIR}/coco_testdev.record
rm -f ${DATASET_DIR}/coco_val.record

# rename the output TF record file to be used for the SSD-ResNet34 model
mv ${DATASET_DIR}/coco_train.record ${DATASET_DIR}/coco_train.record-00000-of-00100

echo "TF records is listed in the dataset directory:"
ls -l ${DATASET_DIR}
