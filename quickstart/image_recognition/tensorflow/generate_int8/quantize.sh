#!/bin/bash
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -x

function main {
  init_params "$@"
  run_tuning

}

# init params
function init_params {

  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --fp32_model=*)
          fp32_model=$(echo $var |cut -f2 -d=)
      ;;
      --int8_model=*)
          int8_model=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    if [ "${topology}" = "inceptionv3" ];then
        conf_yaml=inception_v3.yaml
    elif [ "${topology}" = "resnet50" ]; then
        conf_yaml=resnet50.yaml
    elif [ "${topology}" = "mobilenetv1" ]; then
        conf_yaml=mobilenet_v1.yaml
    elif [ "${topology}" = "resnet101" ]; then
        conf_yaml=resnet101.yaml            
    fi
    sed -i "/\/path\/to\/calibration\/dataset/s|root:.*|root: $dataset_location|g" $conf_yaml
    sed -i "/\/path\/to\/evaluation\/dataset/s|root:.*|root: $dataset_location|g" $conf_yaml

    python quantize_by_INC.py \
            --model ${topology} \
            --config ${conf_yaml} \
            --input-graph ${fp32_model} \
            --output-graph ${int8_model} \
            --tune
}

main "$@"
