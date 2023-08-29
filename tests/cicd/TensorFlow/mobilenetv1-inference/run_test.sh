 #!/bin/bash
set -e

echo "Setup TensorFlow Test Enviroment for MobileNet v1 Inference"

PRECISION=$1
SCRIPT=$2
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/TensorFlow/mobilenetv1-inference/${SCRIPT}/${PRECISION}"}
WORKSPACE=$3
is_lkg_drop=$4
DATASET=$5

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${PRECISION}" == "fp32" || "${PRECISION}" == "bfloat32" ]]; then
  PRETRAINED_MODEL="/tf_dataset/pre-trained-models/mobilenet_v1/fp32/mobilenetv1_fp32_pretrained_model_new.pb"
elif [[ "${PRECISION}" == "int8" ]]; then
  PRETRAINED_MODEL="/tf_dataset/pre-trained-models/mobilenet_v1/int8/mobilenetv1_int8_pretrained_model.pb"
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
OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} PRETRAINED_MODEL=${PRETRAINED_MODEL} DATASET_DIR=${DATASET} ./quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/${SCRIPT}
