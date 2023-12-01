#!/bin/bash
set -e

echo "Setup ITEX-XPU Test Enviroment for MaskRCNN Inference"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/ITEX-XPU/maskrcnn-inference/${PRECISION}"}
WORKSPACE=$2
is_lkg_drop=$3
DATASET=$4

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/tensorflow_setup/bin/activate tensorflow
fi

# run following script
cd models_v2/tensorflow/maskrcnn/inference/gpu
./setup.sh
. ./venv/bin/activate
pushd .
cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN
python scripts/download_weights.py --save_dir=./weights
popd
OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} PRETRAINED_MODEL="DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/weights" DATASET_DIR=${DATASET} ./run_model.sh
cd - 
