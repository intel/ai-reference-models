#!/bin/bash
set -e

echo "Setup PY enivornment"

PY_VERSION=$1
is_lkg_drop=$2
WORKSPACE=$3
AIKIT_RELEASE=$4

if [[ "${is_lkg_drop}" == "true" ]]; then
  if [ ! -d "${WORKSPACE}/pytorch_setup" ]; then
    mkdir -p ${WORKSPACE}/pytorch_setup
    cd ${WORKSPACE}/oneapi_drop_tool
    git submodule update --init --remote --recursive
    python -m pip install -r requirements.txt
    python pdt.py --username=tf_qa_prod --password ${TF_QA_PROD} download --product ipytorch --release ${AIKIT_RELEASE} --part webimage --package-os linux --download-dir ${WORKSPACE}/pytorch_setup
    cd ${WORKSPACE}/pytorch_setup
    chmod +x webimage/l_ipytorch_*offline.sh
    ./webimage/l_ipytorch_*_offline.sh -s -a --install-dir ${WORKSPACE}/pytorch_setup/ --silent --eula accept
  fi
  if [ ! -d "${WORKSPACE}/miniconda3" ]; then
    cd ${WORKSPACE}
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
    rm -rf miniconda3
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh -b -f -p miniconda3
  fi
else
  pip install --upgrade pip
  echo "Installing pytorch"
  pip install intel-extension-for-pytorch==${PY_VERSION}
  python -m pip install torch==$1 torchvision==0.15.1
fi
