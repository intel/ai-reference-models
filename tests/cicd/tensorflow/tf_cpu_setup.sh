#!/bin/bash
set -e

echo "Setup TF enivornment"

FRAMEWORK_VERSION=$1
is_lkg_drop=$2
WORKSPACE=$3
AIKIT_RELEASE=$4

if [[ "${is_lkg_drop}" == "true" ]]; then
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
  echo "Installing tensorflow"
  pip install tensorflow==$1
fi
