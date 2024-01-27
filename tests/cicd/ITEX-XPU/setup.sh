#!/bin/bash
set -e

echo "Setup ITEX-XPU enivornment"

TF_VERSION=$1
TF_EXTENSION_VERSION=$2
is_lkg_drop=$3
WORKSPACE=$4
AIKIT_RELEASE=$5

if [[ "${is_lkg_drop}" == "true" ]]; then
  if [ ! -d "${WORKSPACE}/miniconda3" ]; then
    cd ${WORKSPACE}
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
    rm -rf miniconda3
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh -b -f -p miniconda3
  fi
  rm -rf ${WORKSPACE}/tensorflow_setup
  if [ ! -d "${WORKSPACE}/tensorflow_setup" ]; then
    mkdir -p ${WORKSPACE}/tensorflow_setup
    cd ${WORKSPACE}/oneapi_drop_tool
    git submodule update --init --remote --recursive
    python -m pip install -r requirements.txt
    python cdt.py --username=tf_qa_prod --password ${TF_QA_PROD} download --product tensorflow --release ${AIKIT_RELEASE} -c l_drop_installer --download-dir ${WORKSPACE}/tensorflow_setup
    cd ${WORKSPACE}/tensorflow_setup
    chmod +x ITEX_installer-*
    ./ITEX_installer-* -b -u -p ${WORKSPACE}/tensorflow_setup
  fi
else
  pip install --upgrade pip
  echo "Installing tensorflow and ITEX"
  pip install tensorflow==$1
  pip install --upgrade intel-extension-for-tensorflow[xpu]==$2
fi
