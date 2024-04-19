#
# Copyright (c) 2023 Intel Corporation
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
: ${is_in_container=${1:-"False"}}
: ${base_path=${2:-""}}
: ${output_path=${3:-"/data/mlperf_data"}}
: ${model=${4:-"na"}}
: ${impl=${5:-"na"}}
: ${dtype=${6:-"na"}}

f_output="${output_path}/env_${model}_${impl}_${dtype}.log"

if [ "${is_in_container}" == "False" ]; then
    pushd ${base_path}
    echo " git log --oneline | head -n 1:" > ${f_output}
    git log --oneline | head -n 1 >> ${f_output} || true
    echo "" >> ${f_output}
    popd

    echo "who:" >> ${f_output}
    who >> ${f_output} || true
    echo "" >> ${f_output}
    echo "free -h:" >> ${f_output}
    free -h >> ${f_output} || true
    echo "" >> ${f_output}
    echo "ps -ef | grep python:" >> ${f_output}
    ps -ef | grep python >> ${f_output} || true
    echo "" >> ${f_output}
    echo "lscpu:" >> ${f_output}
    lscpu >> ${f_output} || true
    dmesg | grep "cpu clock throttled" >> ${f_output} || true
    echo "" >> ${f_output}
else
    echo "conda list:" >> ${f_output}
    conda list >> ${f_output} || true
    echo "" >> ${f_output}
fi
