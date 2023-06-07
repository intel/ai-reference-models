 #!/bin/bash
set -e

echo "Setup TensorFlow Test Enviroment for ResNet50 v1.5 Inference"

PRECISION=$1
SCRIPT=$2
DATASET=$3
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/TensorFlow/output/resnet50v1.5-inference/${SCRIPT}/${PRECISION}"}
WORKSPACE=$4
is_lkg_drop=$5

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${PRECISION}" == "bfloat16" ]]; then
  PRETRAINED_MODEL="/tf_dataset/pre-trained-models/resnet50v1_5/bf16/bf16_resnet50_v1.pb"
elif [[ "${PRECISION}" == "fp32" || "${PRECISION}" == "bfloat32" ]]; then
  PRETRAINED_MODEL="/tf_dataset/pre-trained-models/resnet50v1_5/fp32/resnet50_v1.pb"
elif [[ "${PRECISION}" == "int8" ]]; then
  PRETRAINED_MODEL="/tf_dataset/pre-trained-models/resnet50v1_5/int8/bias_resnet50.pb"
fi

if [[ "${is_lkg_drop}" == "true" ]]; then
  export PATH=${WORKSPACE}/miniconda3/bin:$PATH
  source ${WORKSPACE}/tensorflow_setup/setvars.sh
  source ${WORKSPACE}/tensorflow_setup/compiler/latest/env/vars.sh
  source ${WORKSPACE}/tensorflow_setup/mkl/latest/env/vars.sh
  source ${WORKSPACE}/tensorflow_setup/tbb/latest/env/vars.sh
  source ${WORKSPACE}/tensorflow_setup/mpi/latest/env/vars.sh
  conda activate tensorflow
fi

# run following script
OUTPUT_DIR=${OUTPUT_DIR} BATCH_SIZE=${BATCH_SIZE} PRECISION=${PRECISION} PRETRAINED_MODEL=${PRETRAINED_MODEL} DATASET_DIR=${DATASET} ./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/${SCRIPT}
