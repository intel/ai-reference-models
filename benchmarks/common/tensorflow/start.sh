#!/usr/bin/env bash
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

#  inference & training is supported right now
if [ ${MODE} != "inference" ] && [ ${MODE} != "training" ]; then
  echo "${MODE} mode for ${MODEL_NAME} is not supported"
  exit 1
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
else
   echo "$unamestr is not supported!"
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
    if [[ ${OS_PLATFORM} == *"CentOS"* ]] || [[ ${OS_PLATFORM} == *"Red Hat"* ]]; then
      if [[ ! "${OS_VERSION}" =~ "7".* ]] && [[ ! "${OS_VERSION}" =~ "8".* ]]; then
        echo "${OS_PLATFORM} version ${OS_VERSION} is not currently supported."
        exit 1
      fi
    elif [[ ${OS_PLATFORM} == *"Ubuntu"* ]]; then
      if [[ ! "${OS_VERSION}" =~ "18.04".* ]] && [[ ! "${OS_VERSION}" =~ "20.04".* ]]; then
        echo "${OS_PLATFORM} version ${OS_VERSION} is not currently supported."
        exit 1
      fi
    elif [[ ${OS_PLATFORM} == *"Debian"* ]]; then
      if [[ ! "${OS_VERSION}" =~ "10".* ]] && [[ ! "${OS_VERSION}" =~ "11".* ]]; then
        echo "${OS_PLATFORM} version ${OS_VERSION} is not currently supported."
        exit 1
      fi
    elif [[ ${OS_PLATFORM} == *"SLES"* ]]; then
      if [[ ! "${OS_VERSION}" =~ "15".* ]]; then
        echo "${OS_PLATFORM} version ${OS_VERSION} is not currently supported."
        exit 1
      fi
    else
      echo "${OS_PLATFORM} version ${OS_VERSION} is not currently supported."
      exit 1
    fi

    echo "Running on ${OS_PLATFORM} version ${OS_VERSION} is supported."
fi

if [[ ${NOINSTALL} != "True" ]]; then
  # set env var before installs so that user interaction is not required
  export DEBIAN_FRONTEND=noninteractive
  # install common dependencies
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

    if [[ ${MPI_NUM_PROCESSES} != "None" ]]; then
      # Installing OpenMPI
      yum install -y openmpi openmpi-devel openssh openssh-server
      yum clean all
      export PATH="/usr/lib64/openmpi/bin:${PATH}"

      # Install Horovod
      export HOROVOD_WITHOUT_PYTORCH=1
      export HOROVOD_WITHOUT_MXNET=1
      export HOROVOD_WITH_TENSORFLOW=1
      export HOROVOD_VERSION=87094a4

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
      python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
    fi
  elif [[ ${OS_PLATFORM} == *"SLES"* ]]; then
    zypper update -y
    zypper install -y gcc gcc-c++ cmake python3-tk libXext6 libSM6

    # install google-perftools for tcmalloc
    if [[ ${DISABLE_TCMALLOC} != "True" ]]; then
      zypper install -y gperftools && \
      zypper clean all
    fi

    if [[ ${MPI_NUM_PROCESSES} != "None" ]]; then
      ## Installing OpenMPI
      zypper install -y openmpi3 openmpi3-devel openssh openssh-server
      zypper clean all
      export PATH="/usr/lib64/mpi/gcc/openmpi3/bin:${PATH}"

      ## Install Horovod
      export HOROVOD_WITHOUT_PYTORCH=1
      export HOROVOD_WITHOUT_MXNET=1
      export HOROVOD_WITH_TENSORFLOW=1
      export HOROVOD_VERSION=87094a4

      # In case installing released versions of Horovod fail,and there is
      # a working commit replace next set of commands with something like:
      zypper install -y git make
      zypper clean all
      python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
    fi
  elif [[ ${OS_PLATFORM} == *"Ubuntu"* ]] || [[ ${OS_PLATFORM} == *"Debian"* ]]; then
    apt-get update -y
    apt-get install gcc-8 g++-8 cmake python-tk -y
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 700 --slave /usr/bin/g++ g++ /usr/bin/g++-7
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
    apt-get install -y libsm6 libxext6 python3-dev

    # install google-perftools for tcmalloc
    if [[ ${DISABLE_TCMALLOC} != "True" ]]; then
      apt-get install google-perftools -y
    fi

    if [[ ${MPI_NUM_PROCESSES} != "None" ]]; then
      # Installing OpenMPI
      apt-get install openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev -y

      # Install Horovod
      export HOROVOD_WITHOUT_PYTORCH=1
      export HOROVOD_WITHOUT_MXNET=1
      export HOROVOD_WITH_TENSORFLOW=1
      export HOROVOD_VERSION=87094a4

      apt-get update
      # In case installing released versions of Horovod fail,and there is
      # a working commit replace next set of commands with something like:
      apt-get install -y --no-install-recommends --fix-missing cmake git
      python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
      # apt-get install -y --no-install-recommends --fix-missing cmake
      # python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
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
  elif [[ ${OS_PLATFORM} == *"SLES"* ]]; then
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

verbose_arg=""
if [ ${VERBOSE} == "True" ]; then
  verbose_arg="--verbose"
fi

weight_sharing_arg=""
if [ ${WEIGHT_SHARING} == "True" ]; then
  weight_sharing_arg="--weight-sharing"
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
    if [ ${PRECISION} == "fp32" ]; then
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

# MiniGo model
function minigo() {
  if [ ${MODE} == "training" ] && [ ${PRECISION} == "fp32" ]; then
      original_dir=$(pwd)
      local MODEL_DIR=${EXTERNAL_MODELS_SOURCE_DIRECTORY}
      local INTELAI_MODEL_DIR=${INTELAI_MODELS}
      local BENCHMARK_DIR=${BENCHMARK_SCRIPTS}

      if [ -n "${DOCKER}" ]; then
        MODEL_DIR=${MOUNT_EXTERNAL_MODELS_SOURCE}
        INTELAI_MODEL_DIR=${MOUNT_INTELAI_MODELS_SOURCE}
        BENCHMARK_DIR=${MOUNT_BENCHMARK}
        # install dependencies
        apt-get update && apt-get install -y cpio
        # python3 -m pip install -r ${MODEL_DIR}/requirements.txt
        python3 -m pip install -r ${BENCHMARK_DIR}/reinforcement/tensorflow/minigo/requirements.txt
        if [ ! -f "bazel-0.22.0-installer-linux-x86_64.sh" ];then
          wget https://github.com/bazelbuild/bazel/releases/download/0.22.0/bazel-0.22.0-installer-linux-x86_64.sh
          chmod 755 bazel-0.22.0-installer-linux-x86_64.sh
        fi
        ./bazel-0.22.0-installer-linux-x86_64.sh --prefix=/tmp/bazel
        rm /root/.bazelrc
        export PATH=/tmp/bazel/bin:$PATH
        cd /l_mpi
        sh install.sh --silent silent.cfg
        source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
        python3 -m pip install mpi4py
      fi
    if [ ${NOINSTALL} != "True" ]; then
      # install dependencies
      apt-get update && apt-get install -y git
      python3 -m pip install -r ${MOUNT_EXTERNAL_MODELS_SOURCE}/requirements.txt
      python3 -m pip install -r ${BENCHMARK_DIR}/reinforcement/tensorflow/minigo/requirements.txt
      if [ "${EXTERNAL_MODELS_SOURCE_DIRECTORY}" == "None" ]; then
        echo "You are supposed to provide model dir."
        exit 1
      fi

      # MODEL_DIR is the official mlperf minigo repo
      cd ${MODEL_DIR}
      git checkout 60ecb12f29582227a473fdc7cd09c2605f42bcd6

      # delete the previous patch influence
      git reset --hard
      git clean -fd
      rm -rf ./ml_perf/flags/9.mn/

      # remove the quantization tools downloaded before
      rm -rf ${MODEL_DIR}/ml_perf/tools/
      rm -rf ${MODEL_DIR}/cc/ml_perf/tools/

      if [ "${LARGE_SCALE}" == "True" ]; then
        # multi-node mode
        git apply ${INTELAI_MODEL_DIR}/training/fp32/minigo_mlperf_large_scale.patch
        git apply ${INTELAI_MODEL_DIR}/training/fp32/avoid-repeated-clone-multinode.patch
        git apply ${INTELAI_MODEL_DIR}/training/fp32/bazel-clean-large-scale.patch
        # git apply ${INTELAI_MODEL_DIR}/training/fp32/large-scale-no-bg.patch
      elif [ "${LARGE_NUM_CORES}" == "True" ]; then
        # single-node large num mode
        git apply ${INTELAI_MODEL_DIR}/training/fp32/minigo_mlperf.patch
        git apply ${INTELAI_MODEL_DIR}/training/fp32/avoid-repeated-clone-singlenode.patch
        git apply ${INTELAI_MODEL_DIR}/training/fp32/bazel-clean-single-node.patch
        git apply ${INTELAI_MODEL_DIR}/training/fp32/tune_for_many_core.patch
      else
        # single-node normal mode
        git apply ${INTELAI_MODEL_DIR}/training/fp32/minigo_mlperf.patch
        git apply ${INTELAI_MODEL_DIR}/training/fp32/mlperf_split.patch
        git apply ${INTELAI_MODEL_DIR}/training/fp32/avoid-repeated-clone-singlenode.patch
        git apply ${INTELAI_MODEL_DIR}/training/fp32/bazel-clean-single-node.patch
      fi

      # generate the flags with specified iterations
      if [ -z "${STEPS}" ];then
        steps=30
      fi
      mv ml_perf/flags/9/rl_loop.flags ml_perf/flags/9/rl_loop.flags-org
      sed "s/iterations=30/iterations=${STEPS}/g" ml_perf/flags/9/rl_loop.flags-org &> ml_perf/flags/9/rl_loop.flags
      mv ml_perf/flags/9/train.flags ml_perf/flags/9/train.flags-org
      sed "s/train_batch_size=8192/train_batch_size=4096/g" ml_perf/flags/9/train.flags-org &> ml_perf/flags/9/train.flags

      # MiniGo need specified tensorflow version and to build selfplay part with tensorflow c lib.
      rm -rf cc/minigo_tf/tensorflow-*.data
      rm -rf cc/minigo_tf/tensorflow-*.dist-info
      chmod +777 ./cc/configure_tensorflow.sh
      chmod +777 ./build.sh
      ./cc/configure_tensorflow.sh
      pip uninstall -y ./cc/tensorflow_pkg/tensorflow-*.whl
      pip uninstall -y tensorflow
      pip uninstall -y intel-tensorflow
      python3 -m pip install ./cc/tensorflow_pkg/tensorflow-*.whl
      ./build.sh

      # ensure horovod installed
      python3 -m pip install horovod==0.15.1


      # set the python path for quantization tools
      export PYTHONPATH=${PYTHONPATH}:${MODEL_DIR}/cc/ml_perf/tools/api/intel_quantization:${MODEL_DIR}/ml_perf/tools/api/intel_quantization

      # freeze the tfrecord and target to the checkpoint for training
      git apply ${INTELAI_MODEL_DIR}/training/fp32/get-data.patch
      BOARD_SIZE=9 python ml_perf/get_data.py

      # $HOSTLIST.txt contains all the ip address

      if [ ! $multi_node ];then
        unset -v HOSTLIST
      else
        export HOSTLIST=${BENCHMARK_DIR}/node_list
      fi

      cd ${original_dir}
      CMD="${CMD} \
      $(add_arg "--large-scale" ${LARGE_SCALE}) \
      $(add_arg "--num-train-nodes" ${NUM_TRAIN_NODES}) \
      $(add_arg "--num-eval-nodes" ${NUM_EVAL_NODES}) \
      $(add_arg "--quantization" ${QUANTIZATION}) \
      $(add_arg "--multi-node" ${MULTI_NODE})"
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    fi
  else
    echo "MODE=${MODE} PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
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

    # For accuracy, dataset location is required.
    if [ "${DATASET_LOCATION_VOL}" == "None" ] && [ ${ACCURACY_ONLY} == "True" ]; then
      echo "No Data directory specified, accuracy will not be calculated."
      exit 1
    fi

    if [ ${PRECISION} == "int8" ]; then
        CMD="${CMD} $(add_steps_args) $(add_calibration_arg)"
        PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    elif [ ${PRECISION} == "fp32" ] || [ ${PRECISION} == "bfloat16" ]; then
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
      python3 -m pip install ${MOUNT_INTELAI_MODELS_SOURCE}/tensorflow_addons*.whl --no-deps
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

          if [ ${NOINSTALL} != "True" ]; then
            export PYTHONPATH=${PYTHONPATH}:"/workspace/models/research"
            export PYTHONPATH=${PYTHONPATH}:"/workspace/ssd-resnet-benchmarks/scripts/tf_cnn_benchmarks"
          fi

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
          cd ${old_dir}

          CMD="${CMD} \
          $(add_arg "--weight_decay" ${WEIGHT_DECAY}) \
          $(add_arg "--epochs" ${EPOCHS}) \
          $(add_arg "--save_model_steps" ${SAVE_MODEL_STEPS}) \
          $(add_arg "--timeline" ${TIMELINE}) \
          $(add_arg "--num_warmup_batches" ${NUM_WARMUP_BATCHES})"
          local old_pythonpath=${PYTHONPATH}
          export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}:${MOUNT_EXTERNAL_MODELS_SOURCE}/research
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
    #pip install tensorflow-addons==0.6.0  #/workspace/benchmarks/common/tensorflow/tensorflow_addons-0.6.0.dev0-cp36-cp36m-linux_x86_64.whl
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
  if [ ${PRECISION} == "fp32" ]  || [ $PRECISION == "bfloat16" ]; then
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
    # Change if to support fp32
    if [ ${PRECISION} == "fp32" ]  || [ $PRECISION == "int8" ] || [ $PRECISION == "bfloat16" ]; then
      export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
      bert_options
      CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME} in this repo."
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
elif [ ${MODEL_NAME} == "minigo" ]; then
  minigo
elif [ ${MODEL_NAME} == "maskrcnn" ]; then
  maskrcnn
elif [ ${MODEL_NAME} == "mobilenet_v1" ]; then
  mobilenet_v1
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
elif [ ${MODEL_NAME} == "wavenet" ]; then
  wavenet
elif [ ${MODEL_NAME} == "wide_deep" ]; then
  wide_deep
elif [ ${MODEL_NAME} == "wide_deep_large_ds" ]; then
  wide_deep_large_ds
elif [ ${MODEL_NAME} == "bert_base" ]; then
  bert_base
elif [ ${MODEL_NAME} == "bert_large" ]; then
  bert_large
elif [ ${MODEL_NAME} == "dien" ]; then
  dien
else
  echo "Unsupported model: ${MODEL_NAME}"
  exit 1
fi
