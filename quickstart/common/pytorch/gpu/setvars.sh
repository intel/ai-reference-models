#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

if [ -z "${ONEAPI_ROOT}" ]; then
    echo "The ONEAPI_ROOT environment variable was not found. Please source the setvars.sh file before running this script."
    echo "For example: source /opt/intel/oneapi/setvars.sh"
    exit 1
fi

# intel dpcpp compiler
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh

# for oneMKL build specifically
export MKL_DPCPP_ROOT=${ONEAPI_ROOT}/mkl/latest # or version
export LD_LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:$LIBRARY_PATH

# Get the display info from lspci
lspci_display_info=$(lspci | grep -i display)

if [[ ${lspci_display_info} == *"Intel Corporation Device 0bd5"* ]]; then
    export GPU_TYPE="PVC"

    # for AOT
    export USE_AOT_DEVLIST='pvc'

    # HW L2 WA
    export ForceStatelessL1CachingPolicy=1
elif [[ ${lspci_display_info} == *"Intel Corporation Device 020a"* ]]; then
    export GPU_TYPE="ATS"

    # for AOT
    export USE_AOT_DEVLIST='xe_hp_sdv'
else
    echo "Unrecognized GPU type: ${lscpi_display_info}"
    echo "Expected to find Intel Corporation Device 0bd5 (PVC) or 020a (ATS-P) using the command: lspci | grep -i display"
    echo "Please verify the system's configuration and driver setup."
fi

