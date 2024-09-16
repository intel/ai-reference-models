#!/usr/bin/env bash
#
# Copyright (c) 2018-2023 Intel Corporation
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

echo 'Running with parameters:'
echo "    USE_CASE: ${USE_CASE}"
echo "    FRAMEWORK: ${FRAMEWORK}"
echo "    WORKSPACE: ${WORKSPACE}"
echo "    DATASET_LOCATION: ${DATASET_LOCATION}"
echo "    CHECKPOINT_DIRECTORY: ${CHECKPOINT_DIRECTORY}"
echo "    BACKBONE_MODEL_DIRECTORY: ${BACKBONE_MODEL_DIRECTORY}"
echo "    IN_GRAPH: ${IN_GRAPH}"
echo "    MOUNT_INTELAI_MODELS_COMMON_SOURCE_DIR: ${MOUNT_INTELAI_MODELS_COMMON_SOURCE}"
if [ -n "${DOCKER}" ]; then
  echo "    Mounted volumes:"
  echo "        ${BENCHMARK_SCRIPTS} mounted on: ${MOUNT_BENCHMARK}"
  echo "        ${EXTERNAL_MODELS_SOURCE_DIRECTORY} mounted on: ${MOUNT_EXTERNAL_MODELS_SOURCE}"
  echo "        ${INTELAI_MODELS} mounted on: ${MOUNT_INTELAI_MODELS_SOURCE}"
  echo "        ${DATASET_LOCATION_VOL} mounted on: ${DATASET_LOCATION}"
  echo "        ${CHECKPOINT_DIRECTORY_VOL} mounted on: ${CHECKPOINT_DIRECTORY}"
  echo "        ${BACKBONE_MODEL_DIRECTORY_VOL} mounted on: ${BACKBONE_MODEL_DIRECTORY}"
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
echo "    MPI_NUM_PROCESSES: ${MPI_NUM_PROCESSES}"
echo "    MPI_NUM_PEOCESSES_PER_SOCKET: ${MPI_NUM_PROCESSES_PER_SOCKET}"
echo "    MPI_HOSTNAMES: ${MPI_HOSTNAMES}"
echo "    NUMA_CORES_PER_INSTANCE: ${NUMA_CORES_PER_INSTANCE}"
echo "    PYTHON_EXE: ${PYTHON_EXE}"
echo "    PYTHONPATH: ${PYTHONPATH}"
echo "    DRY_RUN: ${DRY_RUN}"
echo "    GPU: ${GPU}"
echo "    ONEDNN_GRAPH: ${ONEDNN_GRAPH}"

#  Enable GPU Flag
gpu_arg=""
is_model_gpu_supported="False"
if [ ${GPU} == "True" ]; then
  gpu_arg="--gpu"
  # Environment variables for GPU
  export RenderCompressedBuffersEnabled=0
  export CreateMultipleSubDevices=1
  export ForceLocalMemoryAccessMode=1
  export SYCL_PI_LEVEL_ZERO_BATCH_SIZE=1
else
  unset RenderCompressedBuffersEnabled
  unset CreateMultipleSubDevices
  unset ForceLocalMemoryAccessMode
  unset ForceNonSystemMemoryPlacement
  unset TF_ENABLE_LAYOUT_OPT
  unset SYCL_PI_LEVEL_ZERO_BATCH_SIZE
fi
#  inference & training is supported right now
if [ ${MODE} != "inference" ] && [ ${MODE} != "training" ]; then
  echo "${MODE} mode for ${MODEL_NAME} is not supported"
  exit 1
fi

# Enable OneDNN Graph Flag
onednn_graph_arg=""
if [ ${ONEDNN_GRAPH} == "True" ]; then
  onednn_graph_arg="--onednn-graph=True"
  export ITEX_ONEDNN_GRAPH=1
fi

# Determines if we are running in a container by checking for .dockerenv
function _running-in-container()
{
  # .dockerenv is a legacy mount populated by Docker engine and at some point it may go away.
  [ -f /.dockerenv ]
}

# check if running on Windows OS
PLATFORM='unknown'
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   PLATFORM='linux'
elif [[ "$unamestr" == "MSYS"* ]]; then
   PLATFORM='windows'
fi
echo
echo "Running on ${PLATFORM}"
echo

OS_PLATFORM=""
if [[ ${PLATFORM} == "linux" ]]; then
    # Check the Linux PLATFORM distribution if CentOS, Debian or Ubuntu
    OS_PLATFORM=$(egrep '^(NAME)=' /etc/os-release)
    OS_PLATFORM=$(echo "${OS_PLATFORM#*=}")
    OS_VERSION=$(egrep '^(VERSION_ID)=' /etc/os-release)
    OS_VERSION=$(echo "${OS_VERSION#*=}")

    echo "Running on ${OS_PLATFORM} version ${OS_VERSION}"
fi

if [[ ${NOINSTALL} != "True" ]]; then
  # set env var before installs so that user interaction is not required
  export DEBIAN_FRONTEND=noninteractive
  # install common dependencies

  # Handle horovod uniformly for all OSs.
  # If a diffferent version need to be used for a specific OS
  # change that variable alone locally in the large if stmts (below)
  if [[ ${MPI_NUM_PROCESSES} != "None"  && $MODE == "training" ]]; then
    export HOROVOD_WITHOUT_PYTORCH=1
    export HOROVOD_WITHOUT_MXNET=1
    export HOROVOD_WITH_TENSORFLOW=1
    export HOROVOD_VERSION=39c8f7c
  fi

  if [[ ${OS_PLATFORM} == *"CentOS"* ]] || [[ ${OS_PLATFORM} == *"Red Hat"* ]]; then
    yum update -y
    yum install -y gcc gcc-c++ cmake python3-tkinter libXext libSM

    # install google-perftools for tcmalloc
    if [[ ${DISABLE_TCMALLOC} != "True" ]]; then
      if [[ ${OS_PLATFORM} == *"Red Hat"* ]] && [[ ${OS_VERSION} =~ "7".* ]]; then
        # For Red Hat 7 we need to build from source
        pushd .
        yum install -y wget

        GPERFTOOLS_VER="2.9.1"
        wget https://github.com/gperftools/gperftools/releases/download/gperftools-${GPERFTOOLS_VER}/gperftools-${GPERFTOOLS_VER}.tar.gz  -O gperftools-${GPERFTOOLS_VER}.tar.gz
        tar -xvzf gperftools-${GPERFTOOLS_VER}.tar.gz
        cd gperftools-${GPERFTOOLS_VER}

        ./configure --disable-cpu-profiler --disable-heap-profiler --disable-heap-checker --disable-debugalloc --enable-minimal
        make
        make install
        LD_LIBRARY_PATH=“/usr/local/lib:${LD_LIBRARY_PATH}”
        popd
      else
        if [[ ${OS_VERSION} =~ "7".* ]]; then
          yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm && \
          yum install -y https://extras.getpagespeed.com/release-el7-latest.rpm
        elif [[ ${OS_VERSION} =~ "8".* ]]; then
          # For Red Hat user needs to register the system first to be able to use the following repositories
          # subscription-manager register --auto-attach
          yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && \
          yum install -y https://extras.getpagespeed.com/release-el8-latest.rpm
        fi
        yum install -y gperftools && \
        yum clean all
      fi
    fi

    if [[ ${MPI_NUM_PROCESSES} != "None"  && $MODE == "training" ]]; then
      # Installing OpenMPI
      yum install -y openmpi openmpi-devel openssh openssh-server
      yum clean all
      export PATH="/usr/lib64/openmpi/bin:${PATH}"

      # Install GCC 7 from devtoolset-7
      if [[ ${OS_VERSION} =~ "7".* ]]; then
        if [[ ${OS_PLATFORM} == *"CentOS"* ]]; then
          yum install -y centos-release-scl
        else
          # For Red Hat user needs to register then enable the repo:
          # subscription-manager repos --enable rhel-7-server-devtools-rpms
          yum install -y scl-utils
        fi
        yum install -y devtoolset-7
        export PATH="/opt/rh/devtoolset-7/root/usr/bin:${PATH}"
      fi

      # In case installing released versions of Horovod fail,and there is
      # a working commit replace next set of commands with something like:
      yum install -y git make
      yum clean all
      CC=gcc CXX=g++ python3 -m pip install --no-cache-dir git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
      horovodrun --check-build
    fi
  elif [[ ${OS_PLATFORM} == *"SLES"* ]] || [[ ${OS_PLATFORM} == *"SUSE"* ]]; then
    zypper update -y
    zypper install -y gcc gcc-c++ cmake python3-tk libXext6 libSM6

    # install google-perftools for tcmalloc
    if [[ ${DISABLE_TCMALLOC} != "True" ]]; then
      zypper install -y gperftools && \
      zypper clean all
    fi

    if [[ ${MPI_NUM_PROCESSES} != "None"  && $MODE == "training" ]]; then
      ## Installing OpenMPI
      zypper install -y openmpi3 openmpi3-devel openssh openssh-server
      zypper clean all
      export PATH="/usr/lib64/mpi/gcc/openmpi3/bin:${PATH}"

      # In case installing released versions of Horovod fail,and there is
      # a working commit replace next set of commands with something like:
      zypper install -y git make
      zypper clean all
      CC=gcc CXX=g++ python3 -m pip install --no-cache-dir git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
      horovodrun --check-build
    fi
  elif [[ ${OS_PLATFORM} == *"Ubuntu"* ]] || [[ ${OS_PLATFORM} == *"Debian"* ]]; then
    apt-get update -y
    apt-get install gcc-9 g++-9 cmake python3-tk -y
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 700 --slave /usr/bin/g++ g++ /usr/bin/g++-7
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 900 --slave /usr/bin/g++ g++ /usr/bin/g++-9
    apt-get install -y libsm6 libxext6 python3-dev

    # install google-perftools for tcmalloc
    if [[ ${DISABLE_TCMALLOC} != "True" ]]; then
      apt-get install google-perftools -y
    fi

    if [[ ${MPI_NUM_PROCESSES} != "None"  && $MODE == "training" ]]; then
      # Installing OpenMPI
      apt-get install openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev -y

      apt-get update
      # In case installing released versions of Horovod fail,and there is
      # a working commit replace next set of commands with something like:
      apt-get install -y --no-install-recommends --fix-missing cmake git
      # TODO: Once this PR https://github.com/horovod/horovod/pull/3864 is merged, we can install horovod as before.
      CC=gcc CXX=g++ python3 -m pip install --no-cache-dir git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}

      # Will keep this as reference for any future usecase
      #git clone https://github.com/horovod/horovod.git
      #cd horovod
      #git reset --hard ${HOROVOD_VERSION}
      #git submodule update --init --recursive
      #git fetch origin pull/3864/head:ashahba/issue-3861-fix
      #git checkout ashahba/issue-3861-fix
      #python3 -m pip install --no-cache-dir -v -e .

      horovodrun --check-build
    fi
  fi
  python3 -m pip install --upgrade 'pip>=20.3.4'
  python3 -m pip install requests
fi

# Determine if numactl needs to be installed
INSTALL_NUMACTL="False"
if [[ $NUMA_CORES_PER_INSTANCE != "None" || $SOCKET_ID != "-1" || $NUM_CORES != "-1" ]]; then
  # The --numa-cores-per-instance, --socket-id, and --num-cores args use numactl
  INSTALL_NUMACTL="True"
elif [[ $MODEL_NAME == "bert_large" && $MODE == "training" && $MPI_NUM_PROCESSES != "None" ]]; then
  # BERT large training with MPI uses numactl
  INSTALL_NUMACTL="True"
fi

# If we are running in a container, call the container_init.sh files
if [[ ${NOINSTALL} != "True" ]]; then
  if _running-in-container ; then
    # For running inside a real CentOS container
    if [[ ${OS_PLATFORM} == *"CentOS"* ]] || [[ ${OS_PLATFORM} == *"Red Hat"* ]]; then
      # Next if block only applies to CentOS 8. Please see here:
      # https://forums.centos.org/viewtopic.php?f=54&t=78708
      if [[ ! ${OS_VERSION} =~ "8".* ]] && [[ ${OS_PLATFORM} != *"Stream"* ]] && [[ ${OS_PLATFORM} != *"Red Hat"* ]]; then
        sed -i '/^mirrorlist=/s/mirrorlist=/#mirrorlist=/g' /etc/yum.repos.d/CentOS-Linux-*
        sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-*
        yum clean all
        yum distro-sync -y
      fi
      if [[ $INSTALL_NUMACTL == "True" ]]; then
        yum update -y
        yum install -y numactl
      fi
    elif [[ ${OS_PLATFORM} == *"SLES"* ]] || [[ ${OS_PLATFORM} == *"SUSE"* ]]; then
      if [[ $INSTALL_NUMACTL == "True" ]]; then
        zypper update -y
        zypper install -y numactl
      fi
    elif [[ ${OS_PLATFORM} == *"Ubuntu"* ]] || [[ ${OS_PLATFORM} == *"Debian"* ]]; then
      # For ubuntu, run the container_init.sh scripts
      if [ -f ${MOUNT_BENCHMARK}/common/${FRAMEWORK}/container_init.sh ]; then
        # Call the framework's container_init.sh, if it exists and we are running on ubuntu
        INSTALL_NUMACTL=$INSTALL_NUMACTL ${MOUNT_BENCHMARK}/common/${FRAMEWORK}/container_init.sh
      fi
      # Call the model specific container_init.sh, if it exists
      if [ -f ${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/${MODE}/${PRECISION}/container_init.sh ]; then
        ${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/${MODE}/${PRECISION}/container_init.sh
      fi
    fi
  fi
fi

verbose_arg=""
if [ ${VERBOSE} == "True" ]; then
  verbose_arg="--verbose"
fi

weight_sharing_arg=""
if [ ${WEIGHT_SHARING} == "True" ]; then
  weight_sharing_arg="--weight-sharing"
fi
synthetic_data_arg=""
if [ ${SYNTHETIC_DATA} == "True" ]; then
  synthetic_data_arg="--synthetic-data"
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

numa_cores_per_instance_arg=""
if [[ -n ${NUMA_CORES_PER_INSTANCE} && ${NUMA_CORES_PER_INSTANCE} != "None" ]]; then
  numa_cores_per_instance_arg="--numa-cores-per-instance=${NUMA_CORES_PER_INSTANCE}"
fi

RUN_SCRIPT_PATH="common/${FRAMEWORK}/run_tf_benchmark.py"

timestamp=`date +%Y%m%d_%H%M%S`
LOG_FILENAME="benchmark_${MODEL_NAME}_${MODE}_${PRECISION}_${timestamp}.log"
if [ ! -d "${OUTPUT_DIR}" ]; then
  mkdir ${OUTPUT_DIR}
fi

export PYTHONPATH=${PYTHONPATH}:${MOUNT_INTELAI_MODELS_COMMON_SOURCE}:${MOUNT_INTELAI_MODELS_SOURCE}

# Common execution command used by all models
function run_model() {
  if [ ${is_model_gpu_supported} == "False"  ] && [ ${GPU} == "True" ]; then
    echo "Runing ${MODEL_NAME} ${MODE} with precision ${PRECISION} does not support --gpu."
    exit 1
  fi
  # Navigate to the main benchmark directory before executing the script,
  # since the scripts use the benchmark/common scripts as well.
  cd ${MOUNT_BENCHMARK}

  # Start benchmarking
  if [[ -z $DRY_RUN ]]; then
    if [[ -z $numa_cores_per_instance_arg ]]; then
      eval ${CMD} 2>&1 | tee ${LOGFILE}
    else
      # Don't tee to a log file for numactl multi-instance runs
      eval ${CMD}
    fi
  else
    echo ${CMD}
    return
  fi

  if [ ${VERBOSE} == "True" ]; then
    echo "PYTHONPATH: ${PYTHONPATH}" | tee -a ${LOGFILE}
    echo "RUNCMD: ${CMD} " | tee -a ${LOGFILE}
    if [[ ${BATCH_SIZE} != "-1" ]]; then
      echo "Batch Size: ${BATCH_SIZE}" | tee -a ${LOGFILE}
    fi
  fi

  if [[ ${BATCH_SIZE} != "-1" ]]; then
    echo "Ran ${MODE} with batch size ${BATCH_SIZE}" | tee -a ${LOGFILE}
  fi

  # if it starts with /workspace then it's not a separate mounted dir
  # so it's custom and is in same spot as LOGFILE is, otherwise it's mounted in a different place
  if [[ "${OUTPUT_DIR}" = "/workspace"* ]]; then
    LOG_LOCATION_OUTSIDE_CONTAINER=${BENCHMARK_SCRIPTS}/common/${FRAMEWORK}/logs/${LOG_FILENAME}
  else
    LOG_LOCATION_OUTSIDE_CONTAINER=${LOGFILE}
  fi

  # Don't print log file location for numactl multi-instance runs, because those have
  # separate log files for each instance
  if [[ -z $numa_cores_per_instance_arg ]]; then
    echo "Log file location: ${LOG_LOCATION_OUTSIDE_CONTAINER}" | tee -a ${LOGFILE}
  fi
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
--num-train-steps=${NUM_TRAIN_STEPS} \
${numa_cores_per_instance_arg} \
${accuracy_only_arg} \
${benchmark_only_arg} \
${output_results_arg} \
${weight_sharing_arg} \
${synthetic_data_arg} \
${verbose_arg} \
${gpu_arg} \
${onednn_graph_arg}"

if [ ${MOUNT_EXTERNAL_MODELS_SOURCE} != "None" ]; then
  CMD="${CMD} --model-source-dir=${MOUNT_EXTERNAL_MODELS_SOURCE}"
fi

if [[ -n "${IN_GRAPH}" && ${IN_GRAPH} != "" ]]; then
  CMD="${CMD} --in-graph=${IN_GRAPH}"
fi

if [[ -n "${CHECKPOINT_DIRECTORY}" && ${CHECKPOINT_DIRECTORY} != "" ]]; then
  CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY}"
fi

if [[ -n "${BACKBONE_MODEL_DIRECTORY}" && ${BACKBONE_MODEL_DIRECTORY} != "" ]]; then
  CMD="${CMD} --backbone-model=${BACKBONE_MODEL_DIRECTORY}"
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

## Added for bert
function bert_options() {

  if [[ ${MODE} == "training" ]]; then
    if [[ -z "${TRAIN_OPTION}" ]]; then
      echo "Error: Please specify a train option (SQuAD, Classifier, Pretraining)"
      exit 1
    fi

    CMD=" ${CMD} --train-option=${TRAIN_OPTION}"
  fi

  if [[ ${MODE} == "inference" ]]; then
    if [[ -z "${INFER_OPTION}" ]]; then
      echo "Error: Please specify a inference option (SQuAD, Classifier, Pretraining)"
      exit 1
    fi

    CMD=" ${CMD} --infer-option=${INFER_OPTION}"
  fi

  if [[ -n "${INIT_CHECKPOINT}" && ${INIT_CHECKPOINT} != "" ]]; then
    CMD=" ${CMD} --init-checkpoint=${INIT_CHECKPOINT}"
  fi

  if [[ -n "${TASK_NAME}" && ${TASK_NAME} != "" ]]; then
    CMD=" ${CMD} --task-name=${TASK_NAME}"
  fi

  if [[ -n "${WARMUP_STEPS}" && ${WARMUP_STEPS} != "" ]]; then
    CMD=" ${CMD} --warmup-steps=${WARMUP_STEPS}"
  fi

  if [[ -n "${STEPS}" && ${STEPS} != "" ]]; then
    CMD=" ${CMD} --steps=${STEPS}"
  fi

  if [[ -n "${VOCAB_FILE}" && ${VOCAB_FILE} != "" ]]; then
    CMD=" ${CMD} --vocab-file=${VOCAB_FILE}"
  fi

  if [[ -n "${CONFIG_FILE}" && ${CONFIG_FILE} != "" ]]; then
    CMD=" ${CMD} --config-file=${CONFIG_FILE}"
  fi

  if [[ -n "${DO_PREDICT}" && ${DO_PREDICT} != "" ]]; then
    CMD=" ${CMD} --do-predict=${DO_PREDICT}"
  fi

  if [[ -n "${PREDICT_FILE}" && ${PREDICT_FILE} != "" ]]; then
    CMD=" ${CMD} --predict-file=${PREDICT_FILE}"
  fi

  if [[ -n "${DO_TRAIN}" && ${DO_TRAIN} != "" ]]; then
    CMD=" ${CMD} --do-train=${DO_TRAIN}"
  fi

  if [[ -n "${TRAIN_FILE}" && ${TRAIN_FILE} != "" ]]; then
    CMD=" ${CMD} --train-file=${TRAIN_FILE}"
  fi

  if [[ -n "${NUM_TRAIN_EPOCHS}" && ${NUM_TRAIN_EPOCHS} != "" ]]; then
    CMD=" ${CMD} --num-train-epochs=${NUM_TRAIN_EPOCHS}"
  fi

  if [[ -n "${NUM_TRAIN_STEPS}" && ${NUM_TRAIN_STEPS} != "" ]]; then
    CMD=" ${CMD} --num-train-steps=${NUM_TRAIN_STEPS}"
  fi

  if [[ -n "${MAX_PREDICTIONS}" && ${MAX_PREDICTIONS} != "" ]]; then
    CMD=" ${CMD} --max-predictions=${MAX_PREDICTIONS}"
  fi

  if [[ -n "${LEARNING_RATE}" && ${LEARNING_RATE} != "" ]]; then
    CMD=" ${CMD} --learning-rate=${LEARNING_RATE}"
  fi

  if [[ -n "${MAX_SEQ_LENGTH}" && ${MAX_SEQ_LENGTH} != "" ]]; then
    CMD=" ${CMD} --max-seq-length=${MAX_SEQ_LENGTH}"
  fi

  if [[ -n "${DOC_STRIDE}" && ${DOC_STRIDE} != "" ]]; then
    CMD=" ${CMD} --doc-stride=${DOC_STRIDE}"
  fi

  if [[ -n "${INPUT_FILE}" && ${INPUT_FILE} != "" ]]; then
    CMD=" ${CMD} --input-file=${INPUT_FILE}"
  fi

  if [[ -n "${DO_EVAL}" && ${DO_EVAL} != "" ]]; then
    CMD=" ${CMD} --do-eval=${DO_EVAL}"
  fi

  if [[ -n "${DATA_DIR}" && ${DATA_DIR} != "" ]]; then
    CMD=" ${CMD} --data-dir=${DATA_DIR}"
  fi

  if [[ -n "${DO_LOWER_CASE}" && ${DO_LOWER_CASE} != "" ]]; then
    CMD=" ${CMD} --do-lower-case=${DO_LOWER_CASE}"
  fi
  if [[ -n "${ACCUM_STEPS}" && ${ACCUM_STEPS} != "" ]]; then
    CMD=" ${CMD} --accum_steps=${ACCUM_STEPS}"
  fi
  if [[ -n "${PROFILE}" && ${PROFILE} != "" ]]; then
    CMD=" ${CMD} --profile=${PROFILE}"
  fi
  if [[ -n "${EXPERIMENTAL_GELU}" && ${EXPERIMENTAL_GELU} != "" ]]; then
    CMD=" ${CMD} --experimental-gelu=${EXPERIMENTAL_GELU}"
  fi
  if [[ -n "${OPTIMIZED_SOFTMAX}" && ${OPTIMIZED_SOFTMAX} != "" ]]; then
    CMD=" ${CMD} --optimized-softmax=${OPTIMIZED_SOFTMAX}"
  fi

  if [[ -n "${MPI_WORKERS_SYNC_GRADIENTS}" && ${MPI_WORKERS_SYNC_GRADIENTS} != "" ]]; then
    CMD=" ${CMD} --mpi_workers_sync_gradients=${MPI_WORKERS_SYNC_GRADIENTS}"
  fi

}

## Added for BERT-large model from HuggingFace
function bert_large_hf_options() {
  # For accuracy, dataset location is required
  if [ "${DATASET_LOCATION_VOL}" == None ]; then
    if [ ${ACCURACY_ONLY} == "True" ]; then
      echo "No dataset directory specified, accuracy cannot be calculated."
      exit 1
    else
      # Download model from huggingface.co/models for benchmarking
      CMD=" ${CMD} --model-name-or-path=bert-large-uncased-whole-word-masking"
    fi
  else
    CMD=" ${CMD} --model-name-or-path=${DATASET_LOCATION_VOL}"
  fi

  if [[ -n "${DATASET_NAME}" && ${DATASET_NAME} != "" ]]; then
    CMD=" ${CMD} --dataset-name=${DATASET_NAME}"
  fi

  if [[ -n "${WARMUP_STEPS}" && ${WARMUP_STEPS} != "" ]]; then
    CMD=" ${CMD} --warmup-steps=${WARMUP_STEPS}"
  fi

  if [[ -n "${STEPS}" && ${STEPS} != "" ]]; then
    CMD=" ${CMD} --steps=${STEPS}"
  fi
}

function install_protoc() {
  pushd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"

  # install protoc, if necessary, then compile protoc files
  if [ ! -f "bin/protoc" ]; then
    install_location=$1
    echo "protoc not found, installing protoc from ${install_location}"
    if [[ ${OS_PLATFORM} == *"CentOS"* ]] || [[ ${OS_PLATFORM} == *"Red Hat"* ]]; then
      yum update -y && yum install -y unzip wget
    else
      apt-get update && apt-get install -y unzip wget
    fi
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
  local trainepochs_arg=""
  local epochsbtweval_arg=""
  local warmup_steps_arg=""
  local kmp_blocktime_arg=""

  if [ -n "${STEPS}" ]; then
    steps_arg="--steps=${STEPS}"
  fi

  if [ -n "${TRAIN_EPOCHS}" ]; then
    trainepochs_arg="--train_epochs=${TRAIN_EPOCHS}"
  fi

  if [ -n "${EPOCHS_BETWEEN_EVALS}" ]; then
    epochsbtweval_arg="--epochs_between_evals=${EPOCHS_BETWEEN_EVALS}"
  fi

  if [ -n "${WARMUP_STEPS}" ]; then
    warmup_steps_arg="--warmup-steps=${WARMUP_STEPS}"
  fi

  if [ -n "${KMP_BLOCKTIME}" ]; then
    kmp_blocktime_arg="--kmp-blocktime=${KMP_BLOCKTIME}"
  fi

  echo "${steps_arg} ${trainepochs_arg} ${epochsbtweval_arg} ${warmup_steps_arg} ${kmp_blocktime_arg}"
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

# 3D UNet model
function 3d_unet() {
  if [[ ${PRECISION} == "fp32" ]] && [[ ${MODE} == "inference" ]]; then
    if [[ ${NOINSTALL} != "True" ]]; then
      python3 -m pip install -r "${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/requirements.txt"
    fi
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_INTELAI_MODELS_SOURCE}/inference/fp32
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "${PRECISION} ${MODE} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# MLPerf 3D UNet model
function 3d_unet_mlperf() {
  # For accuracy, dataset location is required
  # if [ "${DATASET_LOCATION_VOL}" == None ] && [ ${ACCURACY_ONLY} == "True" ]; then
  #   echo "No dataset directory specified, accuracy cannot be calculated."
  #   exit 1
  # fi
  CMD="${CMD} $(add_steps_args)"
  if [ ${MODE} == "inference" ]; then
    if [ ${PRECISION} == "fp32" ]  || [ $PRECISION == "bfloat16" ] || [ $PRECISION == "int8" ]; then
      if [ ${NOINSTALL} != "True" ]; then
        echo "Installing requirements"
        python3 -m pip install -r "${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/requirements.txt"
      fi
      export PYTHONPATH=${PYTHONPATH}:${MOUNT_INTELAI_MODELS_SOURCE}/inference/${PRECISION}
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
      echo "${PRECISION} ${MODE} is not supported for ${MODEL_NAME}"
      exit 1
    fi
  else
    echo "${MODE} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

#BERT model
function bert() {
   if [ ${PRECISION} == "fp32" ]; then
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_BENCHMARK}:${MOUNT_EXTERNAL_MODELS_SOURCE}
    if [ ${NOINSTALL} != "True" ]; then
      apt-get update && apt-get install -y git
      python3 -m pip install -r ${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/requirements.txt
    fi
    CMD="${CMD} \
    $(add_arg "--task_name" ${TASK_NAME}) \
    $(add_arg "--max_seq_length" ${MAX_SEQ_LENGTH}) \
    $(add_arg "--eval_batch_size" ${eval_batch_size}) \
    $(add_arg "--learning_rate" ${LEARNING_RATE}) \
    $(add_arg "--vocab_file" ${VOCAB_FILE}) \
    $(add_arg "--bert_config_file" ${BERT_CONFIG_FILE}) \
    $(add_arg "--init_checkpoint" ${INIT_CHECKPOINT})"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

function dien_options() {
  if [[ -n "${EXACT_MAX_LENGTH}" && ${EXACT_MAX_LENGTH} != "" ]]; then
    CMD=" ${CMD} --exact-max-length=${EXACT_MAX_LENGTH}"
  fi
  if [[ -n "${GRAPH_TYPE}" && ${GRAPH_TYPE} != "" ]]; then
    CMD=" ${CMD} --graph_type=${GRAPH_TYPE}"
  fi
  if [[ -n "${NUM_ITERATIONS}" && ${NUM_ITERATIONS} != "" ]]; then
    CMD=" ${CMD} --num-iterations=${NUM_ITERATIONS}"
  fi
  if [[ -n "${PRECISION}" && ${PRECISION} != "" ]]; then
    CMD=" ${CMD} --data-type=${PRECISION}"
  fi
}

# DIEN model
function dien() {
  if [ ${MODE} == "inference" ]; then
    if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ]; then
      dien_options
      CMD=${CMD} run_model

    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME}"
      exit 1
    fi
  elif [ ${MODE} == "training" ]; then
    if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ]; then
      dien_options
      CMD=${CMD} run_model

    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME}"
      exit 1
    fi
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
      CMD="${CMD} $(add_arg "--input_height" ${INPUT_HEIGHT}) $(add_arg "--input_width" ${INPUT_WIDTH}) \
      $(add_arg "--warmup_steps" ${WARMUP_STEPS}) $(add_arg "--steps" ${STEPS}) $(add_arg "--input_layer" ${INPUT_LAYER}) \
      $(add_arg "--output_layer" ${OUTPUT_LAYER})"
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# Faster R-CNN (ResNet50) model
function faster_rcnn() {
    export PYTHONPATH=$PYTHONPATH:${MOUNT_EXTERNAL_MODELS_SOURCE}/research:${MOUNT_EXTERNAL_MODELS_SOURCE}/research/slim
    original_dir=$(pwd)

    if [ ${NOINSTALL} != "True" ]; then
      # install dependencies
      python3 -m pip install -r "${MOUNT_BENCHMARK}/object_detection/tensorflow/faster_rcnn/requirements.txt"
      cd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"
      # install protoc v3.3.0, if necessary, then compile protoc files
      install_protoc "https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip"

      # Install git so that we can apply the patch
      apt-get update && apt-get install -y git
    fi

    # Apply the patch to the tensorflow/models repo with fixes for the accuracy
    # script and for running with python 3
    cd ${MOUNT_EXTERNAL_MODELS_SOURCE}
    git apply ${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/${PRECISION}/faster_rcnn.patch

    if [ ${PRECISION} == "fp32" ]; then
      if [ -n "${CONFIG_FILE}" ]; then
        CMD="${CMD} --config_file=${CONFIG_FILE}"
      fi

      if [[ -z "${CONFIG_FILE}" ]] && [ ${BENCHMARK_ONLY} == "True" ]; then
        echo "Fast R-CNN requires -- config_file arg to be defined"
        exit 1
      fi

    elif [ ${PRECISION} == "int8" ]; then
      number_of_steps_arg=""
      if [ -n "${NUMBER_OF_STEPS}" ] && [ ${BENCHMARK_ONLY} == "True" ]; then
        CMD="${CMD} --number-of-steps=${NUMBER_OF_STEPS}"
      fi
    else
      echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
      exit 1
    fi
    cd $original_dir
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
}


# inceptionv4 model
function inceptionv4() {
  # For accuracy, dataset location is required
  if [ "${DATASET_LOCATION_VOL}" == None ] && [ ${ACCURACY_ONLY} == "True" ]; then
    echo "No dataset directory specified, accuracy cannot be calculated."
    exit 1
  fi
  # add extra model specific args and then run the model
  CMD="${CMD} $(add_steps_args) $(add_arg "--input-height" ${INPUT_HEIGHT}) \
  $(add_arg "--input-width" ${INPUT_WIDTH}) $(add_arg "--input-layer" ${INPUT_LAYER}) \
  $(add_arg "--output-layer" ${OUTPUT_LAYER})"

  if [ ${PRECISION} == "int8" ]; then
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  elif [ ${PRECISION} == "fp32" ]; then
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
      python3 -m pip install -r ${MOUNT_BENCHMARK}/image_segmentation/tensorflow/maskrcnn/inference/fp32/requirements.txt
    fi
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}:${MOUNT_EXTERNAL_MODELS_SOURCE}/mrcnn
    CMD="${CMD} --data-location=${DATASET_LOCATION}"
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# mobilenet_v1 model
function mobilenet_v1() {
  if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ]; then
    CMD="${CMD} $(add_arg "--input_height" ${INPUT_HEIGHT}) $(add_arg "--input_width" ${INPUT_WIDTH}) \
    $(add_arg "--warmup_steps" ${WARMUP_STEPS}) $(add_arg "--steps" ${STEPS}) \
    $(add_arg "--input_layer" ${INPUT_LAYER}) $(add_arg "--output_layer" ${OUTPUT_LAYER})"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  elif [ ${PRECISION} == "int8" ]; then
    CMD="${CMD} $(add_arg "--input_height" ${INPUT_HEIGHT}) $(add_arg "--input_width" ${INPUT_WIDTH}) \
    $(add_arg "--warmup_steps" ${WARMUP_STEPS}) $(add_arg "--steps" ${STEPS}) \
    $(add_arg "--input_layer" ${INPUT_LAYER}) $(add_arg "--output_layer" ${OUTPUT_LAYER}) \
    $(add_calibration_arg)"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# mobilenet_v2 model
function mobilenet_v2() {
  if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ]; then
    CMD="${CMD} $(add_arg "--input_height" ${INPUT_HEIGHT}) $(add_arg "--input_width" ${INPUT_WIDTH}) \
    $(add_arg "--warmup_steps" ${WARMUP_STEPS}) $(add_arg "--steps" ${STEPS}) \
    $(add_arg "--input_layer" ${INPUT_LAYER}) $(add_arg "--output_layer" ${OUTPUT_LAYER})"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  elif [ ${PRECISION} == "int8" ]; then
    CMD="${CMD} $(add_arg "--input_height" ${INPUT_HEIGHT}) $(add_arg "--input_width" ${INPUT_WIDTH}) \
    $(add_arg "--warmup_steps" ${WARMUP_STEPS}) $(add_arg "--steps" ${STEPS}) \
    $(add_arg "--input_layer" ${INPUT_LAYER}) $(add_arg "--output_layer" ${OUTPUT_LAYER}) \
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
        python3 -m pip install opencv-python
        python3 -m pip install easydict
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
  if [[ -n "${clean}" ]]; then
    CMD="${CMD} --clean"
  fi

  # NCF supports different datasets including ml-1m and ml-20m.
  if [[ -n "${DATASET}" && ${DATASET} != "" ]]; then
    CMD="${CMD} --dataset=${DATASET}"
  fi

  if [[ -n "${TE}" && ${TE} != "" ]]; then
    CMD="${CMD} -te=${TE}"
  fi

  if [ ${PRECISION} == "fp32" -o ${PRECISION} == "bfloat16" ]; then
    # For ncf, if dataset location is empty, script downloads dataset at given location.
    if [ ! -d "${DATASET_LOCATION}" ]; then
      mkdir -p ./dataset
      CMD="${CMD} --data-location=./dataset"
    fi

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

    if [ ${NOINSTALL} != "True" ]; then
      python3 -m pip install -r ${MOUNT_BENCHMARK}/recommendation/tensorflow/ncf/inference/requirements.txt
    fi

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# ResNet101, InceptionV3 model
function resnet101_inceptionv3() {
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

# ResNet50  model
function resnet50() {
    export PYTHONPATH=${PYTHONPATH}:$(pwd):${MOUNT_BENCHMARK}
    is_model_gpu_supported="True"
    if [ ${GPU} == "True" ]; then
      PYTHONPATH=${PYTHONPATH}:${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/gpu
    else
      PYTHONPATH=${PYTHONPATH}:${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/cpu
    fi
    # For accuracy, dataset location is required.
    if [ "${DATASET_LOCATION_VOL}" == "None" ] && [ ${ACCURACY_ONLY} == "True" ]; then
      echo "No Data directory specified, accuracy will not be calculated."
      exit 1
    fi

    if [ ${PRECISION} == "int8" ]; then
        CMD="${CMD} $(add_steps_args) $(add_calibration_arg)"
        PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    elif [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ] || [ ${PRECISION} == "fp16" ]; then
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

    if [ ${NOINSTALL} != "True" ]; then
      # install dependencies
      python3 -m pip install ${MOUNT_INTELAI_MODELS_SOURCE}/tensorflow_addons*.whl --no-deps --force-reinstall
    fi

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
    apt-get update && apt-get install -y git
    # install dependencies
    for line in $(sed 's/#.*//g' ${MOUNT_BENCHMARK}/object_detection/tensorflow/rfcn/requirements.txt)
    do
      python3 -m pip install $line
    done
    original_dir=$(pwd)

    cd ${MOUNT_EXTERNAL_MODELS_SOURCE}
    git apply --ignore-space-change --ignore-whitespace ${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/tf-2.0.patch

    cd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"
    # install protoc v3.3.0, if necessary, then compile protoc files
    install_protoc "https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip"
  fi

  split_arg=""
  if [ -n "${SPLIT}" ] && [ ${ACCURACY_ONLY} == "True" ]; then
      split_arg="--split=${SPLIT}"
  fi

  number_of_steps_arg=""
  if [ -n "${NUMBER_OF_STEPS}" ] && [ ${BENCHMARK_ONLY} == "True" ]; then
      number_of_steps_arg="--number_of_steps=${NUMBER_OF_STEPS}"
  fi
  CMD="${CMD} ${number_of_steps_arg} ${split_arg}"

  cd $original_dir
  PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
}

# SSD-MobileNet model
function ssd_mobilenet() {
  is_model_gpu_supported="True"
  if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ]; then
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
    for line in $(sed 's/#.*//g' ${MOUNT_BENCHMARK}/object_detection/tensorflow/ssd-mobilenet/requirements.txt)
    do
      python3 -m pip install $line
    done
  fi
  CMD="${CMD} $(add_steps_args)"
  CMD="${CMD} $(add_arg "--input-subgraph" ${INPUT_SUBGRAPH})"
  PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
}

# SSD-ResNet34 model
function ssd-resnet34() {
      if [ ${MODE} == "inference" ]; then
        if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ] || [ ${PRECISION} == "int8" ]; then

          old_dir=${PWD}

          if [ ${NOINSTALL} != "True" ]; then
            apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0
            for line in $(sed 's/#.*//g' ${MOUNT_BENCHMARK}/object_detection/tensorflow/ssd-resnet34/requirements.txt)
            do
              python3 -m pip install $line
            done
            model_source_dir=${MOUNT_EXTERNAL_MODELS_SOURCE}
            infer_dir=${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}
          else
            model_source_dir=${EXTERNAL_MODELS_SOURCE_DIRECTORY}
            infer_dir="${INTELAI_MODELS}/${MODE}"
          fi
          benchmarks_patch_path=${infer_dir}/tf_benchmarks.patch
          model_patch_path=${infer_dir}/tensorflow_models_tf2.0.patch

          cd  ${model_source_dir}/../
          cd ssd-resnet-benchmarks
          git apply ${benchmarks_patch_path}

          cd ${model_source_dir}
          git apply ${model_patch_path}
          export PYTHONPATH=${PYTHONPATH}:"/workspace/models/research"
          export PYTHONPATH=${PYTHONPATH}:"/workspace/ssd-resnet-benchmarks/scripts/tf_cnn_benchmarks"

          cd ${old_dir}

          CMD="${CMD} \
          $(add_arg "--warmup-steps" ${WARMUP_STEPS}) \
          $(add_arg "--steps" ${STEPS}) \
          $(add_arg "--input-size" ${INPUT_SIZE})"
          CMD=${CMD} run_model

        else
          echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME}"
          exit 1
        fi
      elif [ ${MODE} == "training" ]; then
        if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ]; then
          if [ ${NOINSTALL} != "True" ]; then
            apt-get update && apt-get install -y cpio git

            # Enter the docker mount directory /l_mpi and install the intel mpi with silent mode
            cd /l_mpi
            sh install.sh --silent silent.cfg
            source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64

            for line in $(sed 's/#.*//g' ${MOUNT_BENCHMARK}/object_detection/tensorflow/ssd-resnet34/requirements.txt)
            do
              python3 -m pip install $line
            done
          fi

          old_dir=${PWD}
          cd /tmp
          rm -rf benchmark_ssd_resnet34
          git clone https://github.com/tensorflow/benchmarks.git benchmark_ssd_resnet34
          cd benchmark_ssd_resnet34
          git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
          git apply ${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/${PRECISION}/tf_benchmarks.patch
          git apply ${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/nhwc-bug-fix.diff
          if [ ${PRECISION} == "bfloat16" ]; then
            git apply ${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/${PRECISION}/benchmark-bfloat16.diff
          fi
          if [ ${SYNTHETIC_DATA} == "True" ]; then
	    git apply ${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/no_gpu_preprocess.diff
          fi
	  cd ${old_dir}

          CMD="${CMD} \
          $(add_arg "--weight_decay" ${WEIGHT_DECAY}) \
          $(add_arg "--epochs" ${EPOCHS}) \
          $(add_arg "--save_model_steps" ${SAVE_MODEL_STEPS}) \
          $(add_arg "--timeline" ${TIMELINE}) \
          $(add_arg "--num_warmup_batches" ${NUM_WARMUP_BATCHES})"
          local old_pythonpath=${PYTHONPATH}
          export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
          export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}/research:"/tmp/benchmark_ssd_resnet34/scripts/tf_cnn_benchmarks"
          CMD=${CMD} run_model
          PYTHONPATH=${old_pythonpath}
        else
          echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME}"
          exit 1
        fi
      fi
}

# SSD-VGG16 model
function ssd_vgg16() {

    if [ ${NOINSTALL} != "True" ]; then
        apt-get update && apt-get install -y git
        python3 -m pip install opencv-python Cython

        if [ ${ACCURACY_ONLY} == "True" ]; then
            # get the python cocoapi
            get_cocoapi ${MOUNT_EXTERNAL_MODELS_SOURCE}/coco ${MOUNT_INTELAI_MODELS_SOURCE}/inference
        fi
    fi

    cp ${MOUNT_INTELAI_MODELS_SOURCE}/__init__.py ${MOUNT_EXTERNAL_MODELS_SOURCE}/dataset
    cp ${MOUNT_INTELAI_MODELS_SOURCE}/__init__.py ${MOUNT_EXTERNAL_MODELS_SOURCE}/preprocessing
    cp ${MOUNT_INTELAI_MODELS_SOURCE}/__init__.py ${MOUNT_EXTERNAL_MODELS_SOURCE}/utility
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
    if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "int8" ]; then

      if [ ${NOINSTALL} != "True" ]; then
        for line in $(sed 's/#.*//g' ${MOUNT_BENCHMARK}/object_detection/tensorflow/ssd-resnet34/requirements.txt)
        do
          python3 -m pip install $line
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
        cd ${MOUNT_EXTERNAL_MODELS_SOURCE}
        git apply ${model_patch_path}

        cd ${old_dir}
      fi
      CMD=${CMD} run_model

    else
        echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
        exit 1
    fi
}

# UNet model
function unet() {
  if [ ${PRECISION} == "fp32" ]; then
    if [[ ${NOINSTALL} != "True" ]]; then
      python3 -m pip install -r "${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/requirements.txt"
    fi

    if [[ -z "${CHECKPOINT_NAME}" ]]; then
      echo "UNet requires -- checkpoint_name arg to be defined"
      exit 1
    fi
    if [ ${ACCURACY_ONLY} == "True" ]; then
      echo "Accuracy testing is not supported for ${MODEL_NAME}"
      exit 1
    fi
    CMD="${CMD} --checkpoint_name=${CHECKPOINT_NAME}"
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# transformer language model from official tensorflow models
function transformer_lt_official() {
  if [ ${PRECISION} == "fp32" ]; then

    if [[ -z "${FILE}" ]]; then
        echo "transformer-language requires -- file arg to be defined"
        exit 1
    fi
    if [[ -z "${FILE_OUT}" ]]; then
        echo "transformer-language requires -- file_out arg to be defined"
        exit 1
    fi
    if [[ -z "${REFERENCE}" ]]; then
        echo "transformer-language requires -- reference arg to be defined"
        exit 1
    fi
    if [[ -z "${VOCAB_FILE}" ]]; then
        echo "transformer-language requires -- vocab_file arg to be defined"
        exit 1
    fi

    if [ ${NOINSTALL} != "True" ]; then
      python3 -m pip install -r "${MOUNT_BENCHMARK}/language_translation/tensorflow/transformer_lt_official/requirements.txt"
    fi

    CMD="${CMD}
    --vocab_file=${DATASET_LOCATION}/${VOCAB_FILE} \
    --file=${DATASET_LOCATION}/${FILE} \
    --file_out=${OUTPUT_DIR}/${FILE_OUT} \
    --reference=${DATASET_LOCATION}/${REFERENCE}"
    PYTHONPATH=${PYTHONPATH}:${MOUNT_BENCHMARK}:${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/${PRECISION}
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# transformer in mlperf Translation for Tensorflow  model
function transformer_mlperf() {
  export PYTHONPATH=${PYTHONPATH}:$(pwd):${MOUNT_BENCHMARK}
  if [[ ${MODE} == "training" ]]; then
    if [[ (${PRECISION} == "bfloat16") || ( ${PRECISION} == "fp32") ]]
    then

      if [[ -z "${RANDOM_SEED}" ]]; then
          echo "transformer-language requires --random_seed arg to be defined"
          exit 1
      fi
      if [[ -z "${PARAMS}" ]]; then
          echo "transformer-language requires --params arg to be defined"
          exit 1
      fi
      if [[ -z "${TRAIN_STEPS}" ]]; then
          echo "transformer-language requires --train_steps arg to be defined"
          exit 1
      fi
      if [[ -z "${STEPS_BETWEEN_EVAL}" ]]; then
          echo "transformer-language requires --steps_between_eval arg to be defined"
          exit 1
      fi
      if [[ -z "${DO_EVAL}" ]]; then
          echo "transformer-language requires --do_eval arg to be defined"
          exit 1
      fi
      if [[ -z "${SAVE_CHECKPOINTS}" ]]; then
          echo "transformer-language requires --save_checkpoints arg to be defined"
          exit 1
      fi
      if [[ -z "${PRINT_ITER}" ]]; then
          echo "transformer-language requires --print_iter arg to be defined"
          exit 1
      fi

      CMD="${CMD} --random_seed=${RANDOM_SEED} --params=${PARAMS} --train_steps=${TRAIN_STEPS} --steps_between_eval=${STEPS_BETWEEN_EVAL} --do_eval=${DO_EVAL} --save_checkpoints=${SAVE_CHECKPOINTS}
      --print_iter=${PRINT_ITER} --save_profile=${SAVE_PROFILE}"
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
      exit 1
    fi
  fi

  if [[ ${MODE} == "inference" ]]; then
    if [[ (${PRECISION} == "bfloat16") || ( ${PRECISION} == "fp32") || ( ${PRECISION} == "int8") ]]; then

      if [[ -z "${PARAMS}" ]]; then
          echo "transformer-language requires --params arg to be defined"
          exit 1
      fi

      if [[ -z "${FILE}" ]]; then
          echo "transformer-language requires -- file arg to be defined"
          exit 1
      fi
      if [[ -z "${FILE_OUT}" ]]; then
          echo "transformer-language requires -- file_out arg to be defined"
          exit 1
      fi
      if [[ -z "${REFERENCE}" ]]; then
          echo "transformer-language requires -- reference arg to be defined"
          exit 1
      fi

      CMD="${CMD} $(add_steps_args) $(add_arg "--params" ${PARAMS}) \
           $(add_arg "--file" ${DATASET_LOCATION}/${FILE}) \
           $(add_arg "--vocab_file" ${DATASET_LOCATION}/${VOCAB_FILE}) \
           $(add_arg "--file_out" ${OUTPUT_DIR}/${FILE_OUT}) \
           $(add_arg "--reference" ${DATASET_LOCATION}/${REFERENCE})"
      echo $CMD

      PYTHONPATH=${PYTHONPATH}:${MOUNT_BENCHMARK}:${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/${PRECISION}
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model

    else
      echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
      exit 1
    fi
  fi
}

# GPT-J base model
function gpt_j() {
    if [ ${MODE} == "inference" ]; then
      if [[ -z "${PRETRAINED_MODEL}" ]]; then
        if [[ ${PRECISION} == "int8" ]]; then
          echo "Need to provided pretrained savedModel for gptj int8"
          exit 1
        fi
      else
        CMD=" ${CMD} --pretrained-model=${PRETRAINED_MODEL}"
      fi
      if [[ (${PRECISION} == "bfloat16") || ( ${PRECISION} == "fp32") || ( ${PRECISION} == "fp16")  || ( ${PRECISION} == "int8")]]; then
        if [[ -z "${CHECKPOINT_DIRECTORY}" ]]; then
          echo "Checkpoint directory not found. The script will download the model."
        else
          export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
          export HF_HOME=${CHECKPOINT_DIRECTORY}
          export HUGGINGFACE_HUB_CACHE=${CHECKPOINT_DIRECTORY}
          export TRANSFORMERS_CACHE=${CHECKPOINT_DIRECTORY}
        fi

        if [ ${BENCHMARK_ONLY} == "True" ]; then
          CMD=" ${CMD} --max_output_tokens=${MAX_OUTPUT_TOKENS}"
          CMD=" ${CMD} --input_tokens=${INPUT_TOKENS}"
          CMD=" ${CMD} --steps=${STEPS}"
          CMD=" ${CMD} --warmup_steps=${WARMUP_STEPS}"
          if [[ -z "${DUMMY_DATA}" ]]; then
            DUMMY_DATA=0
          fi
          CMD=" ${CMD} --dummy_data=${DUMMY_DATA}"
        fi
        CMD=${CMD} run_model
      else
        echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME}."
        exit 1
      fi
    else
      echo "Only inference use-case is supported for now."
      exit 1
    fi
}

# Wavenet model
function wavenet() {
  if [ ${PRECISION} == "fp32" ]; then
    if [[ -z "${CHECKPOINT_NAME}" ]]; then
      echo "wavenet requires -- checkpoint_name arg to be defined"
      exit 1
    fi

    if [[ -z "${SAMPLE}" ]]; then
      echo "wavenet requires -- sample arg to be defined"
      exit 1
    fi

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

    if [ ${NOINSTALL} != "True" ]; then
      python3 -m pip install librosa==0.5
    fi

    CMD="${CMD} --checkpoint_name=${CHECKPOINT_NAME} \
        --sample=${SAMPLE}"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# BERT base
function bert_base() {
  if [ ${GPU} == "True" ]; then
    if [ ${MODE} == "inference" ]; then
      echo "PRECISION=${PRECISION} on GPU not supported for ${MODEL_NAME} ${MODE} in this repo."
      exit 1
    elif [ ${MODE} == "training" ]; then
      if [ ${PRECISION} != "fp32" ] && [ ${PRECISION} != "bfloat16" ]; then
        echo "PRECISION=${PRECISION} on GPU not supported for ${MODEL_NAME} ${MODE} in this repo."
        exit 1
      fi
    fi
    is_model_gpu_supported="True"
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
    bert_options
    CMD=${CMD} run_model
  elif [ ${PRECISION} == "fp32" ]  || [ $PRECISION == "bfloat16" ]; then
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
    bert_options
    CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# BERT Large model
function bert_large() {
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_BENCHMARK}
    if [ ${GPU} == "True" ]; then
      if [ ${MODE} == "inference" ]; then
        if [ ${PRECISION} != "fp32" ] && [ ${PRECISION} != "fp16" ] && [ ${PRECISION} != "bfloat16" ]; then
          echo "PRECISION=${PRECISION} on GPU not supported for ${MODEL_NAME} ${MODE} in this repo."
          exit 1
        fi
      elif [ ${MODE} == "training" ]; then
        if [ ${PRECISION} != "fp32" ] && [ ${PRECISION} != "bfloat16" ]; then
          echo "PRECISION=${PRECISION} on GPU not supported for ${MODEL_NAME} ${MODE} in this repo."
          exit 1
        fi
      fi
      is_model_gpu_supported="True"
      export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
      bert_options
      CMD=${CMD} run_model
    else
      if [ ${PRECISION} == "fp32" ]  || [ $PRECISION == "int8" ] || [ $PRECISION == "bfloat16" ] || [ $PRECISION == "fp16" ]; then
        export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
        bert_options
        CMD=${CMD} run_model
      else
        echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
        exit 1
      fi
    fi
}

# BERT-large model from HuggingFace
function bert_large_hf() {
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_BENCHMARK}
    if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ] || [ ${PRECISION} == "fp16" ]; then
      if [[ ${NOINSTALL} != "True" ]]; then
        python3 -m pip install evaluate git+https://github.com/huggingface/transformers
      fi
      export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
      bert_large_hf_options
      CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
      exit 1
    fi
}

# distilBERT base model
function distilbert_base() {
    if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ]|| [ ${PRECISION} == "int8" ]; then
      export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
      CMD="${CMD} $(add_arg "--warmup-steps" ${WARMUP_STEPS})"
      CMD="${CMD} $(add_arg "--steps" ${STEPS})"

      if [ ${NUM_INTER_THREADS} != "None" ]; then
        CMD="${CMD} $(add_arg "--num-inter-threads" ${NUM_INTER_THREADS})"
      fi

      if [ ${NUM_INTRA_THREADS} != "None" ]; then
        CMD="${CMD} $(add_arg "--num-intra-threads" ${NUM_INTRA_THREADS})"
      fi

      if [ -z ${STEPS} ]; then
        CMD="${CMD} $(add_arg "--steps" ${STEPS})"
      fi

      if [ -z $MAX_SEQ_LENGTH ]; then
        CMD="${CMD} $(add_arg "--max-seq-length" ${MAX_SEQ_LENGTH})"
      fi
      CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
      exit 1
    fi
}

# distilBERT base model
function distilbert_base() {
    if ([ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ] ||
       [ ${PRECISION} == "int8" ] || [ ${PRECISION} == "fp16" ]); then
      export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
      CMD="${CMD} $(add_arg "--warmup-steps" ${WARMUP_STEPS})"
      CMD="${CMD} $(add_arg "--steps" ${STEPS})"

      if [ ${NUM_INTER_THREADS} != "None" ]; then
        CMD="${CMD} $(add_arg "--num-inter-threads" ${NUM_INTER_THREADS})"
      fi

      if [ ${NUM_INTRA_THREADS} != "None" ]; then
        CMD="${CMD} $(add_arg "--num-intra-threads" ${NUM_INTRA_THREADS})"
      fi

      if [ -z ${STEPS} ]; then
        CMD="${CMD} $(add_arg "--steps" ${STEPS})"
      fi

      if [ -z $MAX_SEQ_LENGTH ]; then
        CMD="${CMD} $(add_arg "--max-seq-length" ${MAX_SEQ_LENGTH})"
      fi
      CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
      exit 1
    fi
}

function gpt_j_6B() {
    if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "fp16" ] ||
       [ ${PRECISION} == "bfloat16" ]; then

      if [[ ${INSTALL_TRANSFORMER_FIX} != "True" ]]; then
        echo "Information: Installing transformers from Hugging Face...!"
        echo "python3 -m pip install git+https://github.com/intel-tensorflow/transformers@gptj_add_padding"
        python3 -m pip install git+https://github.com/intel-tensorflow/transformers@gptj_add_padding
      fi

      export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
      CMD="${CMD} $(add_arg "--warmup-steps" ${WARMUP_STEPS})"
      CMD="${CMD} $(add_arg "--steps" ${STEPS})"

      if [[ ${MODE} == "training" ]]; then
        if [[ -z "${TRAIN_OPTION}" ]]; then
          echo "Error: Please specify a train option (GLUE, Lambada)"
          exit 1
        fi

        CMD=" ${CMD} --train-option=${TRAIN_OPTION}"
      fi

      if [[ -z "${CACHE_DIR}" ]]; then
          echo "Checkpoint directory not found. The script will download the model."
      else
          export HF_HOME=${CACHE_DIR}
          export HUGGINGFACE_HUB_CACHE=${CACHE_DIR}
          export TRANSFORMERS_CACHE=${CACHE_DIR}
      fi

      if [ ${NUM_INTER_THREADS} != "None" ]; then
        CMD="${CMD} $(add_arg "--num-inter-threads" ${NUM_INTER_THREADS})"
      fi

      if [ ${NUM_INTRA_THREADS} != "None" ]; then
        CMD="${CMD} $(add_arg "--num-intra-threads" ${NUM_INTRA_THREADS})"
      fi

      if [[ -n "${NUM_TRAIN_EPOCHS}" && ${NUM_TRAIN_EPOCHS} != "" ]]; then
        CMD=" ${CMD} --num-train-epochs=${NUM_TRAIN_EPOCHS}"
      fi

      if [[ -n "${LEARNING_RATE}" && ${LEARNING_RATE} != "" ]]; then
        CMD=" ${CMD} --learning-rate=${LEARNING_RATE}"
      fi

      if [[ -n "${NUM_TRAIN_STEPS}" && ${NUM_TRAIN_STEPS} != "" ]]; then
        CMD=" ${CMD} --num-train-steps=${NUM_TRAIN_STEPS}"
      fi

      if [[ -n "${DO_TRAIN}" && ${DO_TRAIN} != "" ]]; then
        CMD=" ${CMD} --do-train=${DO_TRAIN}"
      fi

      if [[ -n "${DO_EVAL}" && ${DO_EVAL} != "" ]]; then
        CMD=" ${CMD} --do-eval=${DO_EVAL}"
      fi

      if [[ -n "${TASK_NAME}" && ${TASK_NAME} != "" ]]; then
        CMD=" ${CMD} --task-name=${TASK_NAME}"
      fi

      if [[ -n "${CACHE_DIR}" && ${CACHE_DIR} != "" ]]; then
        CMD=" ${CMD} --cache-dir=${CACHE_DIR}"
      fi

      if [[ -n "${PROFILE}" && ${PROFILE} != "" ]]; then
        CMD=" ${CMD} --profile=${PROFILE}"
      fi

      if [ -z ${STEPS} ]; then
        CMD="${CMD} $(add_arg "--steps" ${STEPS})"
      fi

      if [ -z $MAX_SEQ_LENGTH ]; then
        CMD="${CMD} $(add_arg "--max-seq-length" ${MAX_SEQ_LENGTH})"
      fi
      CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
      exit 1
    fi
}


# vision-transformer base model
function vision_transformer() {

    if [ ${MODE} == "training" ]; then
	CMD="${CMD} $(add_arg "--init-checkpoint" ${INIT_CHECKPOINT})"
	CMD="${CMD} $(add_arg "--epochs" ${EPOCHS})"
    fi

    if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ] ||
       [ ${PRECISION} == "fp16" ] || [ ${PRECISION} == "int8" ]; then
      export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
      CMD="${CMD} $(add_arg "--warmup-steps" ${WARMUP_STEPS})"
      CMD="${CMD} $(add_arg "--steps" ${STEPS})"

      if [ ${NUM_INTER_THREADS} != "None" ]; then
        CMD="${CMD} $(add_arg "--num-inter-threads" ${NUM_INTER_THREADS})"
      fi

      if [ ${NUM_INTRA_THREADS} != "None" ]; then
        CMD="${CMD} $(add_arg "--num-intra-threads" ${NUM_INTRA_THREADS})"
      fi

      if [ -z ${STEPS} ]; then
        CMD="${CMD} $(add_arg "--steps" ${STEPS})"
      fi
      CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
      exit 1
    fi
}

# mmoe base model
function mmoe() {
    if [ ${MODE} == "inference" ]; then
      if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ] || [ ${PRECISION} == "fp16" ]; then
        export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
        CMD="${CMD} $(add_arg "--warmup-steps" ${WARMUP_STEPS})"
        CMD="${CMD} $(add_arg "--steps" ${STEPS})"

        if [ ${NUM_INTER_THREADS} != "None" ]; then
          CMD="${CMD} $(add_arg "--num-inter-threads" ${NUM_INTER_THREADS})"
        fi

        if [ ${NUM_INTRA_THREADS} != "None" ]; then
          CMD="${CMD} $(add_arg "--num-intra-threads" ${NUM_INTRA_THREADS})"
        fi

        if [ -z ${STEPS} ]; then
          CMD="${CMD} $(add_arg "--steps" ${STEPS})"
        fi

        CMD=${CMD} run_model
      else
        echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
        exit 1
      fi
    elif [ ${MODE} == "training" ]; then
      if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ] || [ ${PRECISION} == "fp16" ]; then
        export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
        CMD="${CMD} $(add_arg "--train-epochs" ${TRAIN_EPOCHS})"
        CMD="${CMD} $(add_arg "--model_dir" ${CHECKPOINT_DIRECTORY})"
        CMD=${CMD} run_model
      else
        echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
        exit 1
      fi
    fi
}

# rgat base model
function rgat() {
    if [ ${MODE} == "inference" ]; then
      if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ] || [ ${PRECISION} == "fp16" ]; then
        export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

        curr_dir=${pwd}

        infer_dir=${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}
        benchmarks_patch_path=${infer_dir}/tfgnn_legacy_keras.patch
        echo "patch path: $benchmarks_patch_path"

        # Installing tensorflow_gnn from it's main branch
        # python3 -m pip install git+https://github.com/tensorflow/gnn.git@main
        cd /tmp
        rm -rf gnn
        git clone https://github.com/tensorflow/gnn.git
        cd gnn
        git apply $benchmarks_patch_path
        pip install .
        cd ${curr_dir}


        if [ ${NUM_INTER_THREADS} != "None" ]; then
          CMD="${CMD} $(add_arg "--num-inter-threads" ${NUM_INTER_THREADS})"
        fi

        if [ ${NUM_INTRA_THREADS} != "None" ]; then
          CMD="${CMD} $(add_arg "--num-intra-threads" ${NUM_INTRA_THREADS})"
        fi

        CMD="${CMD} $(add_arg "--graph-schema-path" ${GRAPH_SCHEMA_PATH})"
        CMD="${CMD} $(add_arg "--pretrained-model" ${PRETRAINED_MODEL})"
        CMD="${CMD} $(add_arg "--steps" ${STEPS})"
        CMD=${CMD} run_model
      else
        echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
        exit 1
      fi
    fi
}

function stable_diffusion() {
    if [ ${MODE} == "inference" ]; then
      if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ] || [ ${PRECISION} == "fp16" ]; then
        curr_dir=${pwd}

        infer_dir=${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}
        if [[ $TF_USE_LEGACY_KERAS == "1" ]]; then
          benchmarks_patch_path=${infer_dir}/patch_for_stockTF
        else
          benchmarks_patch_path=${infer_dir}/patch
        fi
        echo "patch path: ${benchmarks_patch_path}"

        cd /tmp
        rm -rf keras-cv
        git clone https://github.com/keras-team/keras-cv.git
        cd keras-cv
        git reset --hard 66fa74b6a2a0bb1e563ae8bce66496b118b95200
        git apply ${benchmarks_patch_path}
        pip install .
        cd ${curr_dir}

        if [[ ${NOINSTALL} != "True" ]]; then
          python3 -m pip install -r "${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/${MODE}/requirements.txt"
        fi

        python -c $'from tensorflow import keras\n_ = keras.utils.get_file(
            "bpe_simple_vocab_16e6.txt.gz",
            "https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz?raw=true",
            file_hash="924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
        )\n_ = keras.utils.get_file(
          origin="https://huggingface.co/ianstenbit/keras-sd2.1/resolve/main/text_encoder_v2_1.h5",
          file_hash="985002e68704e1c5c3549de332218e99c5b9b745db7171d5f31fcd9a6089f25b",
        )\n_ = keras.utils.get_file(
          origin="https://huggingface.co/ianstenbit/keras-sd2.1/resolve/main/diffusion_model_v2_1.h5",
          file_hash="c31730e91111f98fe0e2dbde4475d381b5287ebb9672b1821796146a25c5132d",
        )\n_ = keras.utils.get_file(
          origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5",
          file_hash="ad350a65cc8bc4a80c8103367e039a3329b4231c2469a1093869a345f55b1962",
        )'

        export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

        CMD="${CMD} $(add_arg "--steps" ${STEPS})"
        CMD="${CMD} $(add_arg "--output-dir" ${OUTPUT_DIR})"
        CMD=${CMD} run_model
      else
        echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
        exit 1
      fi
    fi
}

# Wide & Deep model
function wide_deep() {
    if [ ${PRECISION} == "fp32" ]; then
      CMD="${CMD} $(add_arg "--pretrained-model" ${PRETRAINED_MODEL})"
      if [ ${NOINSTALL} != "True" ]; then
        echo "Installing requirements"
        python3 -m pip install -r "${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/${MODE}/requirements.txt"
      fi
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

    if [[ -z $LIBTCMALLOC ]] && [[ $NOINSTALL != True ]]; then
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
    if [ ${MODE} == "training" ]; then
      if [[ ! -z ${STEPS} ]]; then
        CMD="${CMD} --steps=${STEPS}"
      fi
      if [ ${PRECISION} == "fp32" ]; then
        CMD="${CMD}"
        PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
      else
        echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME}"
        exit 1
      fi
    fi
    if [ ${MODE} == "inference" ]; then
      if [ "${NUM_OMP_THREADS}" != None ]; then
        CMD="${CMD} --num_omp_threads=${NUM_OMP_THREADS}"
      fi
      if [ "${USE_PARALLEL_BATCHES}" == "True" ]; then
        CMD="${CMD} --use_parallel_batches=${USE_PARALLEL_BATCHES}"
      else
        CMD="${CMD} --use_parallel_batches=False"
      fi
      if [ "${NUM_PARALLEL_BATCHES}" != None  ] && [ "${USE_PARALLEL_BATCHES}" == "True" ]; then
        CMD="${CMD} --num_parallel_batches=${NUM_PARALLEL_BATCHES}"
      fi
      if [ "${KMP_BLOCK_TIME}" != None ] ; then
        CMD="${CMD} --kmp_block_time=${KMP_BLOCK_TIME}"
      fi
      if [ "${KMP_SETTINGS}" != None ]; then
        CMD="${CMD} --kmp_settings=${KMP_SETTINGS}"
      fi
      if [ ${PRECISION} == "int8" ] ||  [ ${PRECISION} == "fp32" ]; then
          CMD="${CMD}"
          PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
      else
          echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
          exit 1
      fi
    fi
}


function edsr() {
  if [ ${PRECISION} == "fp32" ]; then
    CMD="${CMD}  $(add_arg "--warmup_steps" ${WARMUP_STEPS}) $(add_arg "--steps" ${STEPS}) \
    $(add_arg "--input_layer" ${INPUT_LAYER}) $(add_arg "--output_layer" ${OUTPUT_LAYER}) \
    $(add_arg "--use_real_data" ${USE_REAL_DATA})"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi

}

function graphsage() {
    if [ ${MODE} == "inference" ]; then
      if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ] || [ ${PRECISION} == "fp16" ] || [ ${PRECISION} == "int8" ]; then
        export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

        if [ ${NUM_INTER_THREADS} != "None" ]; then
          CMD="${CMD} $(add_arg "--num-inter-threads" ${NUM_INTER_THREADS})"
        fi

        if [ ${NUM_INTRA_THREADS} != "None" ]; then
          CMD="${CMD} $(add_arg "--num-intra-threads" ${NUM_INTRA_THREADS})"
        fi

        CMD="${CMD} $(add_arg "--pretrained-model" ${PRETRAINED_MODEL})"
	CMD="${CMD} $(add_arg "--warmup-steps" ${WARMUP_STEPS})"
        CMD="${CMD} $(add_arg "--steps" ${STEPS})"
        CMD=${CMD} run_model
      else
        echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
        exit 1
      fi
    fi

}

function tiny-yolov4() {
    if [ ${MODE} == "inference" ]; then
      if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ]; then
        export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

        if [ ${NUM_INTER_THREADS} != "None" ]; then
          CMD="${CMD} $(add_arg "--num-inter-threads" ${NUM_INTER_THREADS})"
        fi

        if [ ${NUM_INTRA_THREADS} != "None" ]; then
          CMD="${CMD} $(add_arg "--num-intra-threads" ${NUM_INTRA_THREADS})"
        fi
        CMD=${CMD} run_model
      else
        echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
        exit 1
      fi
    fi
}

function yolov5() {
  if [ ${MODE} == "inference" ] && [ ${BENCHMARK_ONLY} == "True" ]; then
    if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ] || [ ${PRECISION} == "fp16" ] || [ ${PRECISION} == "int8" ]; then
      export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

      if [ ${NUM_INTER_THREADS} != "None" ]; then
        CMD="${CMD} $(add_arg "--num-inter-threads" ${NUM_INTER_THREADS})"
      fi

      if [ ${NUM_INTRA_THREADS} != "None" ]; then
        CMD="${CMD} $(add_arg "--num-intra-threads" ${NUM_INTRA_THREADS})"
      fi
      CMD="${CMD} $(add_arg "--steps" ${STEPS})"
      CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
      exit 1
    fi
  fi
  if [ ${MODE} == "inference" ] && [ ${ACCURACY_ONLY} == "True" ]; then
    if [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ] || [ ${PRECISION} == "fp16" ] || [ ${PRECISION} == "int8" ]; then
      export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
      if [ ${NUM_INTER_THREADS} != "None" ]; then
        CMD="${CMD} $(add_arg "--num-inter-threads" ${NUM_INTER_THREADS})"
      fi

      if [ ${NUM_INTRA_THREADS} != "None" ]; then
        CMD="${CMD} $(add_arg "--num-intra-threads" ${NUM_INTRA_THREADS})"
      fi

      CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
      exit 1
    fi
  fi
}

LOGFILE=${OUTPUT_DIR}/${LOG_FILENAME}

MODEL_NAME=$(echo ${MODEL_NAME} | tr 'A-Z' 'a-z')
if [ ${MODEL_NAME} == "3d_unet" ]; then
  3d_unet
elif [ ${MODEL_NAME} == "3d_unet_mlperf" ]; then
  3d_unet_mlperf
elif [ ${MODEL_NAME} == "bert" ]; then
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
elif [ ${MODEL_NAME} == "mlperf_gnmt" ]; then
  mlperf_gnmt
elif [ ${MODEL_NAME} == "ncf" ]; then
  ncf
elif [ ${MODEL_NAME} == "inceptionv3" ]; then
  resnet101_inceptionv3
elif [ ${MODEL_NAME} == "inceptionv4" ]; then
  inceptionv4
elif [ ${MODEL_NAME} == "maskrcnn" ]; then
  maskrcnn
elif [ ${MODEL_NAME} == "mobilenet_v1" ]; then
  mobilenet_v1
elif [ ${MODEL_NAME} == "mobilenet_v2" ]; then
  mobilenet_v2
elif [ ${MODEL_NAME} == "resnet101" ]; then
  resnet101_inceptionv3
elif [ ${MODEL_NAME} == "resnet50" ]; then
  resnet50
elif [ ${MODEL_NAME} == "resnet50v1_5" ]; then
  resnet50
elif [ ${MODEL_NAME} == "rfcn" ]; then
  rfcn
elif [ ${MODEL_NAME} == "ssd-mobilenet" ]; then
  ssd_mobilenet
elif [ ${MODEL_NAME} == "ssd-resnet34" ]; then
  ssd-resnet34
elif [ ${MODEL_NAME} == "transformer_lt_official" ]; then
  transformer_lt_official
elif [ ${MODEL_NAME} == "transformer_mlperf" ]; then
  transformer_mlperf
elif [ ${MODEL_NAME} == "unet" ]; then
  unet
elif [ ${MODEL_NAME} == "wide_deep" ]; then
  wide_deep
elif [ ${MODEL_NAME} == "wide_deep_large_ds" ]; then
  wide_deep_large_ds
elif [ ${MODEL_NAME} == "bert_base" ]; then
  bert_base
elif [ ${MODEL_NAME} == "bert_large" ]; then
  bert_large
elif [ ${MODEL_NAME} == "bert_large_hf" ]; then
  bert_large_hf
elif [ ${MODEL_NAME} == "dien" ]; then
  dien
elif [ ${MODEL_NAME} == "distilbert_base" ]; then
  distilbert_base
elif [ ${MODEL_NAME} == "vision_transformer" ]; then
  vision_transformer
elif [ ${MODEL_NAME} == "mmoe" ]; then
  mmoe
elif [ ${MODEL_NAME} == "graphsage" ]; then
  graphsage
elif [ ${MODEL_NAME} == "stable_diffusion" ]; then
  stable_diffusion
elif [ ${MODEL_NAME} == "yolov5" ]; then
  yolov5
else
  echo "Unsupported model: ${MODEL_NAME}"
  exit 1
fi
