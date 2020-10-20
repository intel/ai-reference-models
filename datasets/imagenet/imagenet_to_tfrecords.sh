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

# Input args
IMAGENET_HOME=${1}
TPU_REPO=${2}

VALIDATION_TAR=${IMAGENET_HOME}/ILSVRC2012_img_val.tar
TRAIN_TAR=${IMAGENET_HOME}/ILSVRC2012_img_train.tar

# Arg validation: Verify that the IMAGENET_HOME dir exists
if [[ -z ${IMAGENET_HOME} ]]; then
  echo "The directory of ImageNet tar files is required for arg 1"
  exit 1
elif [[ ! -d ${IMAGENET_HOME} ]]; then
  echo "The ImageNet directory (${IMAGENET_HOME}) specified in arg 1 does not exist"
  exit 1
elif [[ ! -f ${VALIDATION_TAR} ]]; then
  echo "The ImageNet validation tar file does not exist at ${VALIDATION_TAR}"
  exit 1
elif [[ ! -f ${TRAIN_TAR} ]]; then
  echo "The ImageNet training tar file does not exist at ${TRAIN_TAR}"
  exit 1
fi

# Arg validation: Verify that the TPU_REPO dir exists
if [[ -z ${TPU_REPO} ]]; then
  echo "The TPU repo is required for arg 2"
  exit 1
elif [[ ! -d ${TPU_REPO} ]]; then
  echo "The TPU repo directory (${TPU_REPO}) specified in arg 2 does not exist"
  exit 1
elif [[ ! -f ${TPU_REPO}/tools/datasets/imagenet_to_gcs.py ]]; then
  echo "The TPU repo ImageNet script was not found at ${TPU_REPO}/tools/datasets/imagenet_to_gcs.py"
  exit 1
fi

# Download the labels file, if it doesn't already exist in the IMAGENET_HOME dir
LABELS_FILE=$IMAGENET_HOME/synset_labels.txt
if [[ ! -f ${LABELS_FILE} ]]; then
  wget -O $LABELS_FILE \
  https://raw.githubusercontent.com/tensorflow/models/v2.3.0/research/inception/inception/data/imagenet_2012_validation_synset_labels.txt
fi

# Setup folders
mkdir -p $IMAGENET_HOME/validation
mkdir -p $IMAGENET_HOME/train
cd $IMAGENET_HOME

# Extract validation and training
tar xf ${VALIDATION_TAR} -C $IMAGENET_HOME/validation
tar xf ${TRAIN_TAR} -C $IMAGENET_HOME/train

# Extract and then delete individual training tar files
cd $IMAGENET_HOME/train
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
  
  # Delete the intermediate tar, since it's no longer needed
  rm $f
done
cd $IMAGENET_HOME # Move back to the base folder

# Navigate to the TPU repo dataset directory and run the conversion script
# The TF Records will end up inthe --local_scratch_dir
cd ${TPU_REPO}/tools/datasets

python3 imagenet_to_gcs.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$IMAGENET_HOME/tf_records \
  --nogcs_upload

# Combine the two folders in tf-records together
cd $IMAGENET_HOME/tf_records
mv train/* $IMAGENET_HOME/tf_records
mv validation/* $IMAGENET_HOME/tf_records
rm -rf train
rm -rf validation
