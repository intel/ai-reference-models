#!/usr/bin/env bash
### This file is originally from: [mlcommons repo](https://github.com/mlcommons/inference/tree/r0.5/others/cloud/single_stage_detector/download_model.sh)
CHECKPOINT_DIR=${CHECKPOINT_DIR-$PWD}

dir=$(pwd)
mkdir -p ${CHECKPOINT_DIR}/ssd; cd ${CHECKPOINT_DIR}/ssd
curl -O https://download.pytorch.org/models/resnet34-333f7ec4.pth
cd $dir
