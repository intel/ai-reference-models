#
<<<<<<<< HEAD:datasets/cloud_data_connector/cloud_data_connector/azure/__init__.py
# -*- coding: utf-8 -*-
#
========
>>>>>>>> r3.1:models_v2/pytorch/bert_large/inference/gpu/setup.sh
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
__path__ = __import__("pkgutil").extend_path(__path__, __name__)  # type: ignore

<<<<<<<< HEAD:datasets/cloud_data_connector/cloud_data_connector/azure/__init__.py
from .connector import Connector
from .downloader import Downloader
from .uploader import Uploader
from .ml_uploader import MLUploader

connect = Connector().connect

__all__=[
    'Connector',
    'Downloader',
    'Uploader',
    'MLUploader'
]
========
# setup.sh
#  - install OS pkgs
#  - should create virtual env & install pip  requirement.txt
#  - git clones & applying patches

set -e
apt-get update && apt-get install -y python3-venv protobuf-compiler

pip install -r requirements.txt

cp -r ../../../../common/parse_result.py common/parse_result.py 
>>>>>>>> r3.1:models_v2/pytorch/bert_large/inference/gpu/setup.sh
