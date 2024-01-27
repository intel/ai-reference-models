#!/bin/bash
set -e

echo "Setup IPEX-XPU enivornment"

FRAMEWORK_VERSION=$1
FRAMEWORK_EXTENSION_VERSION=$2
TORCHVISION_VERSION=$3
is_lkg_drop=$4
AIKIT_RELEASE=$5

if [[ "${is_lkg_drop}" == "true" ]]; then
  if [ ! -d "${GITHUB_WORKSPACE}/miniconda3" ]; then
    cd ${GITHUB_WORKSPACE}
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
    rm -rf miniconda3
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh -b -f -p miniconda3
  fi
  rm -rf ${GITHUB_WORKSPACE}/pytorch_setup
  if [ ! -d "${GITHUB_WORKSPACE}/pytorch_setup" ]; then
    mkdir -p ${GITHUB_WORKSPACE}/pytorch_setup
    cd ${GITHUB_WORKSPACE}/oneapi_drop_tool
    git submodule update --init --remote --recursive
    python -m pip install -r requirements.txt
    python cdt.py --username=tf_qa_prod --password ${TF_QA_PROD} download --product ipytorch --release ${AIKIT_RELEASE} -c l_drop_installer --download-dir ${GITHUB_WORKSPACE}/pytorch_setup
    cd ${GITHUB_WORKSPACE}/pytorch_setup
    chmod +x IPEX_installer-*
    ./IPEX_installer-* -b -u -p ${GITHUB_WORKSPACE}/pytorch_setup
  fi
else
  pip install --upgrade pip
  echo "Installing pytorch"
  export no_proxy=“” 
  export NO_PROXY=“”
  python -m pip install torch==${FRAMEWORK_VERSION} torchvision==${TORCHVISION_VERSION} intel-extension-for-pytorch==${FRAMEWORK_EXTENSION_VERSION} --extra-index-url https://pytorch-extension.intel.com/release-whl-aitools/
fi
