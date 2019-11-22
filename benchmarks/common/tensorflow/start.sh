#!/usr/bin/env bash
#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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
# SPDX-License-Identifier: EPL-2.0
#

echo 'Running with parameters:'
echo "    USE_CASE: ${USE_CASE}"
echo "    FRAMEWORK: ${FRAMEWORK}"
echo "    WORKSPACE: ${WORKSPACE}"
echo "    DATASET_LOCATION: ${DATASET_LOCATION}"
echo "    CHECKPOINT_DIRECTORY: ${CHECKPOINT_DIRECTORY}"
echo "    IN_GRAPH: ${IN_GRAPH}"

if [ ${DOCKER} == "True" ]; then
  echo "    Mounted volumes:"
  echo "        ${BENCHMARK_SCRIPTS} mounted on: ${MOUNT_BENCHMARK}"
  echo "        ${EXTERNAL_MODELS_SOURCE_DIRECTORY} mounted on: ${MOUNT_EXTERNAL_MODELS_SOURCE}"
  echo "        ${INTELAI_MODELS} mounted on: ${MOUNT_INTELAI_MODELS_SOURCE}"
  echo "        ${DATASET_LOCATION_VOL} mounted on: ${DATASET_LOCATION}"
  echo "        ${CHECKPOINT_DIRECTORY_VOL} mounted on: ${CHECKPOINT_DIRECTORY}"
fi

echo "    SOCKET_ID: ${SOCKET_ID}"
echo "    MODEL_NAME: ${MODEL_NAME}"
echo "    MODE: ${MODE}"
echo "    PRECISION: ${PRECISION}"
echo "    BATCH_SIZE: ${BATCH_SIZE}"
echo "    NUM_CORES: ${NUM_CORES}"
echo "    BENCHMARK_ONLY: ${BENCHMARK_ONLY}"
echo "    ACCURACY_ONLY: ${ACCURACY_ONLY}"
echo "    OUTPUT_RESULTS: ${OUTPUT_RESULTS}"
echo "    DISABLE_TCMALLOC: ${DISABLE_TCMALLOC}"
echo "    TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD: ${TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD}"
echo "    NOINSTALL: ${NOINSTALL}"
echo "    OUTPUT_DIR: ${OUTPUT_DIR}"

# Only inference is supported right now
if [ ${MODE} != "inference" ]; then
  echo "${MODE} mode is not supported"
  exit 1
fi

if [[ ${NOINSTALL} != "True" ]]; then
  ## install common dependencies
  apt update
  apt full-upgrade -y
  # Set env var before installs so that user interaction is not required
  export DEBIAN_FRONTEND=noninteractive
  apt-get install python-tk numactl -y
  apt install -y libsm6 libxext6
  pip install --upgrade pip
  pip install requests

  # install google-perftools for tcmalloc
  if [[ ${DISABLE_TCMALLOC} != "True" ]]; then
    apt-get install google-perftools -y
  fi
fi

verbose_arg=""
if [ ${VERBOSE} == "True" ]; then
  verbose_arg="--verbose"
fi

accuracy_only_arg=""
if [ ${ACCURACY_ONLY} == "True" ]; then
  accuracy_only_arg="--accuracy-only"
fi

benchmark_only_arg=""
if [ ${BENCHMARK_ONLY} == "True" ]; then
  benchmark_only_arg="--benchmark-only"
fi

output_results_arg=""
if [ ${OUTPUT_RESULTS} == "True" ]; then
  output_results_arg="--output-results"
fi

RUN_SCRIPT_PATH="common/${FRAMEWORK}/run_tf_benchmark.py"

timestamp=`date +%Y%m%d_%H%M%S`
LOG_FILENAME="benchmark_${MODEL_NAME}_${MODE}_${PRECISION}_${timestamp}.log"
if [ ! -d "${OUTPUT_DIR}" ]; then
  mkdir ${OUTPUT_DIR}
fi

export PYTHONPATH=${PYTHONPATH}:${MOUNT_INTELAI_MODELS_SOURCE}

# Common execution command used by all models
function run_model() {
  # Navigate to the main benchmark directory before executing the script,
  # since the scripts use the benchmark/common scripts as well.
  cd ${MOUNT_BENCHMARK}

  # Start benchmarking
  eval ${CMD} 2>&1 | tee ${LOGFILE}

  if [ ${VERBOSE} == "True" ]; then
    echo "PYTHONPATH: ${PYTHONPATH}" | tee -a ${LOGFILE}
    echo "RUNCMD: ${CMD} " | tee -a ${LOGFILE}
    echo "Batch Size: ${BATCH_SIZE}" | tee -a ${LOGFILE}
  fi
  echo "Ran ${MODE} with batch size ${BATCH_SIZE}" | tee -a ${LOGFILE}

  # if it starts with /workspace then it's not a separate mounted dir
  # so it's custom and is in same spot as LOGFILE is, otherwise it's mounted in a different place
  if [[ "${OUTPUT_DIR}" = "/workspace"* ]]; then
    LOG_LOCATION_OUTSIDE_CONTAINER=${BENCHMARK_SCRIPTS}/common/${FRAMEWORK}/logs/${LOG_FILENAME}
  else
    LOG_LOCATION_OUTSIDE_CONTAINER=${LOGFILE}
  fi
  echo "Log location outside container: ${LOG_LOCATION_OUTSIDE_CONTAINER}" | tee -a ${LOGFILE}
}

# basic run command with commonly used args
CMD="${PYTHON_EXE} ${RUN_SCRIPT_PATH} \
--framework=${FRAMEWORK} \
--use-case=${USE_CASE} \
--model-name=${MODEL_NAME} \
--precision=${PRECISION} \
--mode=${MODE} \
--benchmark-dir=${MOUNT_BENCHMARK} \
--intelai-models=${MOUNT_INTELAI_MODELS_SOURCE} \
--num-cores=${NUM_CORES} \
--batch-size=${BATCH_SIZE} \
--socket-id=${SOCKET_ID} \
--output-dir=${OUTPUT_DIR} \
${accuracy_only_arg} \
${benchmark_only_arg} \
${output_results_arg} \
${verbose_arg}"

if [ ${MOUNT_EXTERNAL_MODELS_SOURCE} != "None" ]; then
  CMD="${CMD} --model-source-dir=${MOUNT_EXTERNAL_MODELS_SOURCE}"
fi

if [[ -n "${IN_GRAPH}" && ${IN_GRAPH} != "" ]]; then
  CMD="${CMD} --in-graph=${IN_GRAPH}"
fi

if [[ -n "${CHECKPOINT_DIRECTORY}" && ${CHECKPOINT_DIRECTORY} != "" ]]; then
  CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY}"
fi

if [[ -n "${DATASET_LOCATION}" && ${DATASET_LOCATION} != "" ]]; then
  CMD="${CMD} --data-location=${DATASET_LOCATION}"
fi

if [ ${NUM_INTER_THREADS} != "None" ]; then
  CMD="${CMD} --num-inter-threads=${NUM_INTER_THREADS}"
fi

if [ ${NUM_INTRA_THREADS} != "None" ]; then
  CMD="${CMD} --num-intra-threads=${NUM_INTRA_THREADS}"
fi

if [ ${DATA_NUM_INTER_THREADS} != "None" ]; then
  CMD="${CMD} --data-num-inter-threads=${DATA_NUM_INTER_THREADS}"
fi

if [ ${DATA_NUM_INTRA_THREADS} != "None" ]; then
  CMD="${CMD} --data-num-intra-threads=${DATA_NUM_INTRA_THREADS}"
fi

if [ ${DISABLE_TCMALLOC} != "None" ]; then
  CMD="${CMD} --disable-tcmalloc=${DISABLE_TCMALLOC}"
fi

function install_protoc() {
  pushd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"

  # install protoc, if necessary, then compile protoc files
  if [ ! -f "bin/protoc" ]; then
    install_location=$1
    echo "protoc not found, installing protoc from ${install_location}"
    apt-get -y install wget
    wget -O protobuf.zip ${install_location}
    unzip -o protobuf.zip
    rm protobuf.zip
  else
    echo "protoc already found"
  fi

  echo "Compiling protoc files"
  ./bin/protoc object_detection/protos/*.proto --python_out=.
  popd
}

function get_cocoapi() {
  # get arg for where the cocoapi repo was cloned
  cocoapi_dir=${1}

  # get arg for the location where we want the pycocotools
  parent_dir=${2}
  pycocotools_dir=${parent_dir}/pycocotools

  # If pycoco tools aren't already found, then builds the coco python API
  if [ ! -d ${pycocotools_dir} ]; then
    # This requires that the cocoapi is cloned in the external model source dir
    if [ -d "${cocoapi_dir}/PythonAPI" ]; then
      # install cocoapi
      pushd ${cocoapi_dir}/PythonAPI
      echo "Installing COCO API"
      make
      cp -r pycocotools ${parent_dir}
      popd
    else
      echo "${cocoapi_dir}/PythonAPI directory was not found"
      echo "Unable to install the python cocoapi."
      exit 1
    fi
  else
    echo "pycocotools were found at: ${pycocotools_dir}"
  fi
}

function add_arg() {
  local arg_str=""
  if [ -n "${2}" ]; then
    arg_str=" ${1}=${2}"
  fi
  echo "${arg_str}"
}

function add_steps_args() {
  # returns string with --steps and --warmup_steps, if there are values specified
  local steps_arg=""
  local warmup_steps_arg=""
  local kmp_blocktime_arg=""

  if [ -n "${steps}" ]; then
    steps_arg="--steps=${steps}"
  fi

  if [ -n "${warmup_steps}" ]; then
    warmup_steps_arg="--warmup-steps=${warmup_steps}"
  fi

  if [ -n "${kmp_blocktime}" ]; then
    kmp_blocktime_arg="--kmp-blocktime=${kmp_blocktime}"
  fi

  echo "${steps_arg} ${warmup_steps_arg} ${kmp_blocktime_arg}"
}

function add_calibration_arg() {
  # returns string with --calibration-only, if True is specified,
  # in this case a subset (~ 100 images) of the ImageNet dataset
  # is generated to be used later on in calibrating the Int8 model.
  # also this function returns a string with --calibrate, if True is specified,
  # which enables resnet50 Int8 benchmark to run accuracy using the previously
  # generated ImageNet data subset.
  local calibration_arg=""

  if [[ ${calibration_only} == "True" ]]; then
    calibration_arg="--calibration-only"
  elif [[ ${calibrate} == "True" ]]; then
    calibration_arg="--calibrate=True"
  fi

  echo "${calibration_arg}"
}

#BERT model
function bert() {
   if [ ${PRECISION} == "fp32" ]; then
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_BENCHMARK}:${MOUNT_EXTERNAL_MODELS_SOURCE}

    if [ ${NOINSTALL} != "True" ]; then
      apt-get update && apt-get install -y git
      pip install -r ${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/requirements.txt
    fi

    CMD="${CMD} \
    $(add_arg "--task_name" ${task_name}) \
    $(add_arg "--max_seq_length" ${max_seq_length}) \
    $(add_arg "--eval_batch_size" ${eval_batch_size}) \
    $(add_arg "--learning_rate" ${learning_rate}) \
    $(add_arg "--vocab_file" ${vocab_file}) \
    $(add_arg "--bert_config_file" ${bert_config_file}) \
    $(add_arg "--init_checkpoint" ${init_checkpoint})"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# DCGAN model
function dcgan() {
  if [ ${PRECISION} == "fp32" ]; then

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}/research:${MOUNT_EXTERNAL_MODELS_SOURCE}/research/slim:${MOUNT_EXTERNAL_MODELS_SOURCE}/research/gan/cifar

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# DenseNet 169 model
function densenet169() {
  if [ ${PRECISION} == "fp32" ]; then
      CMD="${CMD} $(add_arg "--input_height" ${input_height}) $(add_arg "--input_width" ${input_width}) \
      $(add_arg "--warmup_steps" ${warmup_steps}) $(add_arg "--steps" ${steps}) $(add_arg "--input_layer" ${input_layer}) \
      $(add_arg "--output_layer" ${output_layer})"
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# DRAW model
function draw() {
  if [ ${PRECISION} == "fp32" ]; then
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# FaceNet model
function facenet() {
  if [ ${PRECISION} == "fp32" ]; then
    cp ${MOUNT_INTELAI_MODELS_SOURCE}/${PRECISION}/validate_on_lfw.py \
        ${MOUNT_EXTERNAL_MODELS_SOURCE}/src/validate_on_lfw.py

    CMD="${CMD} $(add_arg "--lfw_pairs" ${lfw_pairs})"
    PYTHONPATH=${PYTHONPATH}:${MOUNT_BENCHMARK}:${MOUNT_EXTERNAL_MODELS_SOURCE}
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit
  fi
}

# Faster R-CNN (ResNet50) model
function faster_rcnn() {
    export PYTHONPATH=$PYTHONPATH:${MOUNT_EXTERNAL_MODELS_SOURCE}/research:${MOUNT_EXTERNAL_MODELS_SOURCE}/research/slim
    original_dir=$(pwd)

    if [ ${NOINSTALL} != "True" ]; then
      # install dependencies
      pip install -r "${MOUNT_BENCHMARK}/object_detection/tensorflow/faster_rcnn/requirements.txt"
      cd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"
      # install protoc v3.3.0, if necessary, then compile protoc files
      install_protoc "https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip"

      # install cocoapi
      get_cocoapi ${MOUNT_EXTERNAL_MODELS_SOURCE}/cocoapi ${MOUNT_EXTERNAL_MODELS_SOURCE}/research/
    fi

    if [ ${PRECISION} == "fp32" ]; then
      if [ -n "${config_file}" ]; then
        CMD="${CMD} --config_file=${config_file}"
      fi

      if [[ -z "${config_file}" ]] && [ ${BENCHMARK_ONLY} == "True" ]; then
        echo "Fast R-CNN requires -- config_file arg to be defined"
        exit 1
      fi

    elif [ ${PRECISION} == "int8" ]; then
      number_of_steps_arg=""
      if [ -n "${number_of_steps}" ] && [ ${BENCHMARK_ONLY} == "True" ]; then
        CMD="${CMD} --number-of-steps=${number_of_steps}"
      fi
    else
      echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
      exit 1
    fi
    cd $original_dir
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
}

# GNMT model
function gnmt() {
    export PYTHONPATH=${PYTHONPATH}:$(pwd):${MOUNT_BENCHMARK}

    if [ ${PRECISION} == "fp32" ]; then

      CMD="${CMD} $(add_arg "--src" ${src}) $(add_arg "--tgt" ${tgt}) $(add_arg "--hparams_path" ${hparams_path}) \
      $(add_arg "--vocab_prefix" ${vocab_prefix}) $(add_arg "--inference_input_file" ${inference_input_file}) \
      $(add_arg "--inference_output_file" ${inference_output_file}) $(add_arg "--inference_ref_file" ${inference_ref_file}) \
      $(add_arg "--infer_mode" ${infer_mode})"
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME}"
      exit 1
    fi
}

# inceptionv4 model
function inceptionv4() {
  # For accuracy, dataset location is required
  if [ "${DATASET_LOCATION_VOL}" == None ] && [ ${ACCURACY_ONLY} == "True" ]; then
    echo "No dataset directory specified, accuracy cannot be calculated."
    exit 1
  fi
  # add extra model specific args and then run the model
  CMD="${CMD} $(add_steps_args) $(add_arg "--input-height" ${input_height}) \
  $(add_arg "--input-width" ${input_width}) $(add_arg "--input-layer" ${input_layer}) \
  $(add_arg "--output-layer" ${output_layer})"

  if [ ${PRECISION} == "int8" ]; then
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  elif [ ${PRECISION} == "fp32" ]; then
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# inception_resnet_v2 model
function inception_resnet_v2() {
  # For accuracy, dataset location is required, see README for more information.
  if [ "${DATASET_LOCATION_VOL}" == None ] && [ ${ACCURACY_ONLY} == "True" ]; then
    echo "No Data directory specified, accuracy will not be calculated."
    exit 1
  fi

  if [ ${PRECISION} == "int8" ] || [ ${PRECISION} == "fp32" ]; then
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# language modeling lm-1b
function lm-1b() {
  if [ ${PRECISION} == "fp32" ]; then
    CMD="${CMD} $(add_steps_args)"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# Mask R-CNN model
function maskrcnn() {
  if [ ${PRECISION} == "fp32" ]; then
    original_dir=$(pwd)

    if [ ${NOINSTALL} != "True" ]; then
      # install dependencies
      pip3 install -r ${MOUNT_EXTERNAL_MODELS_SOURCE}/requirements.txt

      # install cocoapi
      get_cocoapi ${MOUNT_EXTERNAL_MODELS_SOURCE}/coco ${MOUNT_EXTERNAL_MODELS_SOURCE}/samples/coco
    fi
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}:${MOUNT_EXTERNAL_MODELS_SOURCE}/samples/coco:${MOUNT_EXTERNAL_MODELS_SOURCE}/mrcnn
    cd ${original_dir}
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# mobilenet_v1 model
function mobilenet_v1() {
  if [ ${PRECISION} == "fp32" ]; then
    CMD="${CMD} $(add_arg "--input_height" ${input_height}) $(add_arg "--input_width" ${input_width}) \
    $(add_arg "--warmup_steps" ${warmup_steps}) $(add_arg "--steps" ${steps}) \
    $(add_arg "--input_layer" ${input_layer}) $(add_arg "--output_layer" ${output_layer})"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  elif [ ${PRECISION} == "int8" ]; then
    CMD="${CMD} $(add_arg "--input_height" ${input_height}) $(add_arg "--input_width" ${input_width}) \
    $(add_arg "--warmup_steps" ${warmup_steps}) $(add_arg "--steps" ${steps}) \
    $(add_arg "--input_layer" ${input_layer}) $(add_arg "--output_layer" ${output_layer}) \
    $(add_calibration_arg)"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# MTCC model
function mtcc() {
  if [ ${PRECISION} == "fp32" ]; then
    if [ ! -d "${DATASET_LOCATION}" ]; then
      echo "No Data location specified, please follow MTCC README instaructions to download the dataset."
      exit 1
    fi
    if [ ${NOINSTALL} != "True" ]; then
      # install dependencies
        pip install opencv-python
        pip install easydict
    fi
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}:${MOUNT_EXTERNAL_MODELS_SOURCE}/Detection:${MOUNT_INTELAI_MODELS_SOURCE}/inference/fp32:${MOUNT_INTELAI_MODELS_SOURCE}/inference/fp32/Detection

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# NCF model
function ncf() {
  if [ ${PRECISION} == "fp32" ]; then
    # For nfc, if dataset location is empty, script downloads dataset at given location.
    if [ ! -d "${DATASET_LOCATION}" ]; then
      mkdir -p /dataset
    fi

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

    if [ ${NOINSTALL} != "True" ]; then
      pip install -r ${MOUNT_EXTERNAL_MODELS_SOURCE}/official/requirements.txt
    fi

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# ResNet50, ResNet101, InceptionV3 model
function resnet50_101_inceptionv3() {
    export PYTHONPATH=${PYTHONPATH}:$(pwd):${MOUNT_BENCHMARK}

    # For accuracy, dataset location is required.
    if [ "${DATASET_LOCATION_VOL}" == "None" ] && [ ${ACCURACY_ONLY} == "True" ]; then
      echo "No Data directory specified, accuracy will not be calculated."
      exit 1
    fi

    if [ ${PRECISION} == "int8" ]; then
        CMD="${CMD} $(add_steps_args) $(add_calibration_arg)"
        PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    elif [ ${PRECISION} == "fp32" ]; then
      CMD="${CMD} $(add_steps_args)"
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
      exit 1
    fi
}

# MLPerf GNMT model
function mlperf_gnmt() {
    export PYTHONPATH=${PYTHONPATH}:$(pwd):${MOUNT_BENCHMARK}

    # install dependencies
    pip install ${MOUNT_BENCHMARK}/tensorflow_addons*.whl --no-deps

    # For accuracy, dataset location is required.
    if [ "${DATASET_LOCATION_VOL}" == "None" ] && [ ${ACCURACY_ONLY} == "True" ]; then
      echo "No Data directory specified, accuracy will not be calculated."
      exit 1
    fi

    if [ ${PRECISION} == "int8" ]; then
      CMD="${CMD} $(add_steps_args) $(add_calibration_arg)"
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    elif [ ${PRECISION} == "fp32" ]; then
      CMD="${CMD} $(add_steps_args)"
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
      exit 1
    fi
}

# R-FCN (ResNet101) model
function rfcn() {
  export PYTHONPATH=$PYTHONPATH:${MOUNT_EXTERNAL_MODELS_SOURCE}/research:${MOUNT_EXTERNAL_MODELS_SOURCE}/research/slim:${MOUNT_EXTERNAL_MODELS_SOURCE}

  if [ ${NOINSTALL} != "True" ]; then
    # install dependencies
    for line in $(cat ${MOUNT_BENCHMARK}/object_detection/tensorflow/rfcn/requirements.txt)
    do
      pip install $line
    done
    original_dir=$(pwd)

    cd ${MOUNT_EXTERNAL_MODELS_SOURCE}
    git apply --ignore-space-change --ignore-whitespace ${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/tf-2.0.patch

    cd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"
    # install protoc v3.3.0, if necessary, then compile protoc files
    install_protoc "https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip"
  fi

  split_arg=""
  if [ -n "${split}" ] && [ ${ACCURACY_ONLY} == "True" ]; then
      split_arg="--split=${split}"
  fi

  number_of_steps_arg=""
  if [ -n "${number_of_steps}" ] && [ ${BENCHMARK_ONLY} == "True" ]; then
      number_of_steps_arg="--number_of_steps=${number_of_steps}"
  fi
  CMD="${CMD} ${number_of_steps_arg} ${split_arg}"

  cd $original_dir
  PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
}

# SqueezeNet model
function squeezenet() {
  if [ ${PRECISION} == "fp32" ]; then
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# SSD-MobileNet model
function ssd_mobilenet() {
  if [ ${PRECISION} == "fp32" ]; then
    if [ ${BATCH_SIZE} != "-1" ]; then
      echo "Warning: SSD-MobileNet FP32 inference script does not use the batch_size arg"
    fi
  elif [ ${PRECISION} != "int8" ]; then
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi

  export PYTHONPATH=${PYTHONPATH}:${MOUNT_BENCHMARK}

  if [ ${NOINSTALL} != "True" ]; then
    # install dependencies for both fp32 and int8
    apt-get update && apt-get install -y git
    # install one by one to solve dependency problems
    for line in $(cat ${MOUNT_BENCHMARK}/object_detection/tensorflow/ssd-mobilenet/requirements.txt)
    do
      pip install $line
    done
  fi

  PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
}

# SSD-ResNet34 model
function ssd-resnet34() {
    if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "int8" ]; then
      if [ ${NOINSTALL} != "True" ]; then
        for line in $(cat ${MOUNT_BENCHMARK}/object_detection/tensorflow/ssd-resnet34/requirements.txt)
        do
          pip install $line
        done
        
        old_dir=${PWD}
        
        infer_dir=${MOUNT_INTELAI_MODELS_SOURCE}/inference
        benchmarks_patch_path=${infer_dir}/tensorflow_benchmarks_tf2.0.patch
        cd /tmp
        git clone --single-branch https://github.com/tensorflow/benchmarks.git
        cd benchmarks
        git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
        git apply ${benchmarks_patch_path}
        
        model_patch_path=${infer_dir}/tensorflow_models_tf2.0.patch
        cd /workspace/models
        git apply ${model_patch_path}
        
        cd ${old_dir}
      fi

      CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME}"
      exit 1
    fi
}

# SSD-VGG16 model
function ssd_vgg16() {

    if [ ${NOINSTALL} != "True" ]; then
        pip install opencv-python Cython

        if [ ${ACCURACY_ONLY} == "True" ]; then
            # get the python cocoapi
            get_cocoapi ${MOUNT_EXTERNAL_MODELS_SOURCE}/coco ${MOUNT_INTELAI_MODELS_SOURCE}/inference
        fi
    fi

    cp ${MOUNT_INTELAI_MODELS_SOURCE}/__init__.py ${MOUNT_EXTERNAL_MODELS_SOURCE}/dataset
    cp ${MOUNT_INTELAI_MODELS_SOURCE}/__init__.py ${MOUNT_EXTERNAL_MODELS_SOURCE}/preprocessing
    cp ${MOUNT_INTELAI_MODELS_SOURCE}/__init__.py ${MOUNT_EXTERNAL_MODELS_SOURCE}/utility
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

    if [ ${PRECISION} == "int8" ] || [ ${PRECISION} == "fp32" ]; then
       CMD="${CMD} $(add_steps_args)"
       PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
        echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
        exit 1
    fi
}

# UNet model
function unet() {
  if [ ${PRECISION} == "fp32" ]; then
    if [[ -z "${checkpoint_name}" ]]; then
      echo "wavenet requires -- checkpoint_name arg to be defined"
      exit 1
    fi
    if [ ${ACCURACY_ONLY} == "True" ]; then
      echo "Accuracy testing is not supported for ${MODEL_NAME}"
      exit 1
    fi
    CMD="${CMD} --checkpoint_name=${checkpoint_name}"
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# transformer language model
function transformer_language() {
  if [ ${PRECISION} == "fp32" ]; then

    if [[ -z "${decode_from_file}" ]]; then
        echo "transformer-language requires -- decode_from_file arg to be defined"
        exit 1
    fi
    if [[ -z "${CHECKPOINT_DIRECTORY}" ]]; then
        echo "transformer-language requires --checkpoint arg to be defined"
        exit 1
    fi
    if [[ -z "${DATASET_LOCATION}" ]]; then
        echo "transformer-language requires --data-location arg to be defined"
        exit 1
    fi

    if [ ${NOINSTALL} != "True" ]; then
      # install dependencies
      echo "Installing tensor2tensor for CPU..."
      pip install tensor2tensor[tensorflow]
    fi

    cp ${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/${PRECISION}/decoding.py ${MOUNT_EXTERNAL_MODELS_SOURCE}/tensor2tensor/utils/decoding.py

    CMD="${CMD} --decode_from_file=${CHECKPOINT_DIRECTORY}/${decode_from_file}"

    if [[ -n "${reference}" ]]; then
      CMD="${CMD} --reference=${CHECKPOINT_DIRECTORY}/${reference}"
    fi

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# transformer language model from official tensorflow models
function transformer_lt_official() {
  if [ ${PRECISION} == "fp32" ]; then

    if [[ -z "${file}" ]]; then
        echo "transformer-language requires -- file arg to be defined"
        exit 1
    fi
    if [[ -z "${file_out}" ]]; then
        echo "transformer-language requires -- file_out arg to be defined"
        exit 1
    fi
    if [[ -z "${reference}" ]]; then
        echo "transformer-language requires -- reference arg to be defined"
        exit 1
    fi
    if [[ -z "${vocab_file}" ]]; then
        echo "transformer-language requires -- vocab_file arg to be defined"
        exit 1
    fi

    cp ${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/${PRECISION}/infer_ab.py \
        ${MOUNT_EXTERNAL_MODELS_SOURCE}/official/transformer/infer_ab.py

    CMD="${CMD}
    --in_graph=${IN_GRAPH} \
    --vocab_file=${DATASET_LOCATION}/${vocab_file} \
    --file=${DATASET_LOCATION}/${file} \
    --file_out=${OUTPUT_DIR}/${file_out} \
    --reference=${DATASET_LOCATION}/${reference}"
    PYTHONPATH=${PYTHONPATH}:${MOUNT_BENCHMARK}:${MOUNT_EXTERNAL_MODELS_SOURCE}
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# Wavenet model
function wavenet() {
  if [ ${PRECISION} == "fp32" ]; then
    if [[ -z "${checkpoint_name}" ]]; then
      echo "wavenet requires -- checkpoint_name arg to be defined"
      exit 1
    fi

    if [[ -z "${sample}" ]]; then
      echo "wavenet requires -- sample arg to be defined"
      exit 1
    fi

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

    if [ ${NOINSTALL} != "True" ]; then
      pip install -r ${MOUNT_EXTERNAL_MODELS_SOURCE}/requirements.txt
    fi

    CMD="${CMD} --checkpoint_name=${checkpoint_name} \
        --sample=${sample}"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# Wide & Deep model
function wide_deep() {
    if [ ${PRECISION} == "fp32" ]; then
      export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

      CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME}"
      exit 1
    fi
}

# Wide & Deep large dataset model
function wide_deep_large_ds() {
    export PYTHONPATH=${PYTHONPATH}:$(pwd):${MOUNT_BENCHMARK}

    # Depends on the Ubuntu version the ldpreload gets installed on various places.
    # Hence getting the best available one from ldconfig and setting it up

    TCMALLOC_LIB="libtcmalloc.so.4"
    LIBTCMALLOC="$(ldconfig -p | grep $TCMALLOC_LIB | tr ' ' '\n' | grep /)"

    if [[ -z "${LIBTCMALLOC}" ]]; then
      echo "libtcmalloc.so.4 not found, trying to install"
      apt-get update
      apt-get install google-perftools --fix-missing -y
    fi

    LIBTCMALLOC="$(ldconfig -p | grep $TCMALLOC_LIB | tr ' ' '\n' | grep /)"
    echo $LIBTCMALLOC
    export LD_PRELOAD=$LIBTCMALLOC
    if [[ -z "${LIBTCMALLOC}" ]]; then
      echo "Failed to load $TCMALLOC_LIB"
    fi

    # Dataset file is required, see README for more information.
    if [ "${DATASET_LOCATION_VOL}" == None ]; then
      echo "Wide & Deep requires --data-location arg to be defined"
      exit 1
    fi

    num_parallel_batches_arg=""

    if [ -n "${num_parallel_batches}" ]; then
      num_parallel_batches_arg="--num-parallel-batches=${num_parallel_batches}"
    fi

    if [ ${PRECISION} == "int8" ] ||  [ ${PRECISION} == "fp32" ]; then
        CMD="${CMD} ${num_parallel_batches_arg} "
        PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
        echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
        exit 1
    fi
}

LOGFILE=${OUTPUT_DIR}/${LOG_FILENAME}
echo "Log output location: ${LOGFILE}"

MODEL_NAME=$(echo ${MODEL_NAME} | tr 'A-Z' 'a-z')
if [ ${MODEL_NAME} == "bert" ]; then
  bert
elif [ ${MODEL_NAME} == "dcgan" ]; then
  dcgan
elif [ ${MODEL_NAME} == "densenet169" ]; then
  densenet169
elif [ ${MODEL_NAME} == "draw" ]; then
  draw
elif [ ${MODEL_NAME} == "facenet" ]; then
  facenet
elif [ ${MODEL_NAME} == "faster_rcnn" ]; then
  faster_rcnn
elif [ ${MODEL_NAME} == "gnmt" ]; then
  gnmt
elif [ ${MODEL_NAME} == "mlperf_gnmt" ]; then
  mlperf_gnmt
elif [ ${MODEL_NAME} == "inceptionv3" ]; then
  resnet50_101_inceptionv3
elif [ ${MODEL_NAME} == "inceptionv4" ]; then
  inceptionv4
elif [ ${MODEL_NAME} == "inception_resnet_v2" ]; then
  inception_resnet_v2
elif [ ${MODEL_NAME} == "lm-1b" ]; then
  lm-1b
elif [ ${MODEL_NAME} == "maskrcnn" ]; then
  maskrcnn
elif [ ${MODEL_NAME} == "mobilenet_v1" ]; then
  mobilenet_v1
elif [ ${MODEL_NAME} == "mtcc" ]; then
  mtcc
elif [ ${MODEL_NAME} == "ncf" ]; then
  ncf
elif [ ${MODEL_NAME} == "resnet101" ]; then
  resnet50_101_inceptionv3
elif [ ${MODEL_NAME} == "resnet50" ]; then
  resnet50_101_inceptionv3
elif [ ${MODEL_NAME} == "resnet50v1_5" ]; then
  resnet50_101_inceptionv3
elif [ ${MODEL_NAME} == "rfcn" ]; then
  rfcn
elif [ ${MODEL_NAME} == "squeezenet" ]; then
  squeezenet
elif [ ${MODEL_NAME} == "ssd-mobilenet" ]; then
  ssd_mobilenet
elif [ ${MODEL_NAME} == "ssd-resnet34" ]; then
  ssd-resnet34
elif [ ${MODEL_NAME} == "ssd_vgg16" ]; then
  ssd_vgg16
elif [ ${MODEL_NAME} == "unet" ]; then
  unet
elif [ ${MODEL_NAME} == "transformer_language" ]; then
  transformer_language
elif [ ${MODEL_NAME} == "transformer_lt_official" ]; then
  transformer_lt_official
elif [ ${MODEL_NAME} == "wavenet" ]; then
  wavenet
elif [ ${MODEL_NAME} == "wide_deep" ]; then
  wide_deep
elif [ ${MODEL_NAME} == "wide_deep_large_ds" ]; then
  wide_deep_large_ds
else
  echo "Unsupported model: ${MODEL_NAME}"
  exit 1
fi
