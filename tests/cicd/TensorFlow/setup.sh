#!/bin/bash
set -e

echo "Setup TF enivornment"

TF_VERSION=$1
is_lkg_drop=$2
WORKSPACE=$3

if [[ "${is_lkg_drop}" == "true" ]]; then
  cd ${WORKSPACE}
  if [ ! -d "${WORKSPACE}/tensorflow_setup" ]; then
    wget --user tf_qa_prod --password=${TF_QA_PROD} https://ubit-artifactory-or.intel.com/artifactory/satgoneapi-or-local/products/tensorflow/2023.1.1/packages/l_tensorflow_p_2023.1.1.48868/webimage/l_tensorflow_p_2023.1.1.48868_offline.sh
    chmod +x l_tensorflow_p_2023.1.1.48868_offline.sh
    mkdir -p tensorflow_setup
    ./l_tensorflow_p_2023.1.1.48868_offline.sh -s -a --install-dir ${WORKSPACE}/tensorflow_setup/ --silent --eula accept
  fi
  if [ ! -d "${WORKSPACE}/miniconda3" ]; then
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
    rm -rf miniconda3
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh -b -f -p miniconda3
  fi
else
  pip install --upgrade pip
  echo "Installing tensorflow"
  pip install intel-tensorflow==$1
fi
