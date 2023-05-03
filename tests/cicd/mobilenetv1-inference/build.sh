#!/bin/bash
set -e

echo "Setup TensorFlow Test Enviroment for MobileNet v1 Inference"

BATCH_SIZE=$1
PRECISION=$2
SCRIPT=$3
DATASET=$5
MODEL_DIR=${MODEL_DIR-"$(pwd)"}
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/output/mobilenetv1-inference/${PRECISION}"}

echo "Setup TensorFlow Test Enviroment..."
# env setup
VENV_DIR="${MODEL_ZOO_REPO_ROOT}/venv"
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv ${VENV_DIR}
fi
source ${VENV_DIR}/bin/activate

${VENV_DIR}/bin/python3 -m pip install --upgrade pip
echo "Installing tensorflow"
pip install intel-tensorflow==$4

if [[ $2 == "fp32" ]];then
  PRETRAINED_MODEL="/tf_dataset/pre-trained-models/mobilenet_v1/fp32/mobilenetv1_fp32_pretrained_model_new.pb"
elif [ "$2" = "int8" ]; then
  PRETRAINED_MODEL="/tf_dataset/pre-trained-models/mobilenet_v1/int8/mobilenetv1_int8_pretrained_model_new.pb"
elif [ "$2" = "bfloat16" ];then
  PRETRAINED_MODEL="/tf_dataset/pre-trained-models/mobilenet_v1/fp32/mobilenetv1_fp32_pretrained_model_new.pb"
else
  echo "No pretrained model is available for the specificied precision $2"
fi

echo "Running MobileNet V1 ${SCRIPT} for ${PRECISION} precision ..."
# run following script
cd ../../..
OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} PRETRAINED_MODEL=${PRETRAINED_MODEL} DATASET_DIR=${DATASET} ./quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/$3
cd -

deactivate
rm -r ${VENV_DIR}
