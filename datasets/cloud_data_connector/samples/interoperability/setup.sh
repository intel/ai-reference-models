#
# -*- coding: utf-8 -*-
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
cp ../../cloud_data_connector/gcp/README.md ./GCP_README.md
cp ../../cloud_data_connector/azure/AzureML.md ./AZUREML_README.md
cp ../../cloud_data_connector/aws/README.md ./AWS_README.md
cp ../azure/sample_data/credit_card_clients.xls ./credit_card_clients.xls
cp ../azure/.env.sample .env.sample
cp ../azure/config.json.sample config.json.sample
cp -r ../azure/src src
cp -r ../azure/dependencies dependencies
python3 -m pip install virtualenv
cd ../../../../
python3 -m virtualenv .venv_intel
. .venv_intel/bin/activate
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install ipython==8.12.0
python3 -m pip install build
cd datasets/cloud_data_connector
python3 -m build .
cd dist/
pip install cloud_data_connector-1.0.2-py3-none-any.whl
cd ..
sudo apt-get update -y && sudo apt-get install -y curl git vim
# Download and run the Poetry installation script
curl -sSL https://install.python-poetry.org | python3 -
poetry install --file samples/interoperability/pyproject.toml
cd samples/interoperability
ipython kernel install --user --name=intel_sample_env1
jupyter notebook Interoperability.ipynb

