#!/bin/bash
set -e

echo "Setup ITEX-XPU Test Enviroment for MaskRCNN Training"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/output/ITEX-XPU/maskrcnn-training/${PRECISION}"}
is_lkg_drop=$2
DATASET=$3

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/tensorflow_setup/bin/activate tensorflow
else
  source /oneapi/compiler/latest/env/vars.sh
  source /oneapi/mpi/latest/env/vars.sh
  source /oneapi/mkl/latest/env/vars.sh
  source /oneapi/tbb/latest/env/vars.sh
  source /oneapi/ccl/latest/env/vars.sh
fi

# run following script
cd models_v2/tensorflow/maskrcnn/training/gpu
./setup.sh
pushd .
cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN
python scripts/download_weights.py --save_dir=./weights
popd
pip install intel-optimization-for-horovod
OUTPUT_DIR=${OUTPUT_DIR} PRECISION=${PRECISION} DATASET_DIR=${DATASET} MULTI_TILE=False ./run_model.sh
cd - 
