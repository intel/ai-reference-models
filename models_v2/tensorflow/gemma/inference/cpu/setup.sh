#!/usr/bin/env bash
#
# Copyright (c) 2024 Intel Corporation
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

set -e
pip install keras
pip install keras-nlp
pip install numpy==1.26.4
if [[ $JAX_NIGHTLY == "1" ]]; then
    echo "Installing JAX nightly"
    pip install -U --pre jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
else
    echo "Installing JAX release version"
    pip install jax
fi
