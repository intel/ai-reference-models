#!/bin/bash
set -e

echo "Setup PyTorch enivornment"

FRAMEWORK_VERSION=$1
is_lkg_drop=$2
AIKIT_RELEASE=$3
WORKSPACE=$4

if [[ "${is_lkg_drop}" == "true" ]]; then
  rm -rf ${WORKSPACE}/pytorch_setup
  if [ ! -d "${WORKSPACE}/pytorch_setup" ]; then
    mkdir -p ${WORKSPACE}/pytorch_setup
    cd ${WORKSPACE}/oneapi_drop_tool
    git submodule update --init --remote --recursive
    python -m pip install -r requirements.txt
    python cdt.py --username=tf_qa_prod --password ${TF_QA_PROD} download --product ipytorch --release ${AIKIT_RELEASE} -c l_drop_installer --download-dir ${WORKSPACE}/pytorch_setup
    cd ${WORKSPACE}/pytorch_setup
    chmod +x IPEX_installer-*
    ./IPEX_installer-* -b -u -p ${WORKSPACE}/pytorch_setup
  fi
else
  pip install --upgrade pip
  echo "Installing pytorch"
  export no_proxy=“”
  export NO_PROXY=“”
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  python -m pip install intel-extension-for-pytorch
  python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
fi

# Check the operating system type
os_type=$(awk -F= '/^NAME/{print $2}' /etc/os-release)

# Install model specific dependencies:
if [[ "$os_name" == *"CentOS"* ]]; then
  echo "CentOS detected. Using yum for package management."
  yum update -y
  yum install -y \
    ca-certificates \
    git \
    cmake>=3.19.6 \
    make \
    autoconf \
    bzip2 \
    tar
  yum install -y \
    numactl \
    mesa-libGL
  yum install -y libsndfile
  yum clean all
  yum install mesa-libGL
elif [[ "$os_name" == *"Ubuntu"* ]]; then
  echo "Ubuntu detected. Using apt-get for package management."
  apt-get update
  apt-get install --no-install-recommends --fix-missing -y \
    build-essential \
    ca-certificates \
    git \
    wget \
    make \
    cmake \
    autoconf \
    bzip2 \
    tar
  apt-get install cmake
  apt-get install --no-install-recommends --fix-missing -y \
    numactl \
    libgl1 \
    libglib2.0-0 \
    libegl1-mesa
  apt-get install -y python3-dev
  apt-get install -y gcc python3.10-dev
  apt-get install -y libgl1-mesa-glx
fi

cd ${WORKSPACE}
if [ -d "jemalloc" ]; then
  echo "Repository already exists. Skipping clone."
else
  unset LD_PRELOAD
  unset MALLOC_CONF
  git clone https://github.com/jemalloc/jemalloc.git
  cd jemalloc
  git checkout c8209150f9d219a137412b06431c9d52839c7272
  ./autogen.sh
  ./configure --prefix=${WORKSPACE}/
  make
  make install
  cd -
fi

pip install packaging intel-openmp

if [ -d "gperftools-2.7.90" ]; then
  echo "The gperftools directory exists. Skipping download and extraction."
else
  wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz
  tar -xzf gperftools-2.7.90.tar.gz
  cd gperftools-2.7.90
  ./configure --prefix=${WORKSPACE}/tcmalloc
  make
  make install
  cd -
fi
