#!/bin/bash
set -e

echo "Setup TensorFlow Test Enviroment for ResNet50 v1.5 Inference"

BATCH_SIZE=$1
PRECISION=$2
SCRIPT=$3
DATASET=$5
MODEL_DIR=${MODEL_DIR-"$(pwd)"}
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/output/resnet50v1.5-inference/${PRECISION}"}
PRETRAINED_MODEL=${PRETRAINED_MODEL-"/tf_dataset/pre-trained-models/resnet50v1_5/"${PRECISION}"/resnet50_v1.pb"}

# env setup
VENV_DIR="${MODEL_DIR}/venv"
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv ${VENV_DIR}
fi
source ${VENV_DIR}/bin/activate

${VENV_DIR}/bin/python3 -m pip install --upgrade pip
echo "Installing tensorflow"
pip install intel-tensorflow==$4

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ ! -f "${PRETRAINED_MODEL}" ]]; then
  echo "Downloading the pretrianed model..."
  wget -O resnet50_v1.pb https://zenodo.org/record/2535873/files/resnet50_v1.pb
  PRETRAINED_MODEL=$(pwd)"/model"
  mv resnet50_v1.pb ${PRETRAINED_MODEL}
  PRETRAINED_MODEL=${PRETRAINED_MODEL-"resnet50_v1.pb"}
fi

# run following script
cd ../../..
OUTPUT_DIR=${OUTPUT_DIR} BATCH_SIZE=${BATCH_SIZE} PRECISION=${PRECISION} PRETRAINED_MODEL=${PRETRAINED_MODEL} DATASET_DIR=${DATASET} ./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/${SCRIPT}
cd -

deactivate
rm -r ${VENV_DIR}
