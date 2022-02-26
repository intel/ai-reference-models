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

import json
import os
import pandas as pd

from zipfile import ZipFile


def get_model_map(json_path, return_data_frame=False):
    """
    Gets the model map from the speified json path and loads it into a python dictionary. If the
    data frame option is enabled, it will also return the list of models in a pandas data frame
    with column headers so that it can be used to display in a notebook.
    """
    with open(json_path) as json_file:
        tfhub_model_map = json.load(json_file)

    if return_data_frame:
        # Generate list of model names and URL links to TF Hub based on the model map
        model_options = [[i,
                          tfhub_model_map[i]["num_hidden_layers"],
                          tfhub_model_map[i]["hidden_size"],
                          tfhub_model_map[i]["num_attention_heads"],
                          "<a href=\"{0}\" target=\"_blank\">{0}</a>".format(
                              tfhub_model_map[i]["bert_encoder"])]
                         for i in tfhub_model_map.keys()]

        if len(model_options) == 0:
            print("Warning: No models were found in the json file:", json_path)

        pd.set_option('display.max_colwidth', None)
        models_df = pd.DataFrame(model_options,
                                 columns=["Model",
                                          "Hidden layers",
                                          "Hidden size",
                                          "Attention heads",
                                          "TF Hub BERT encoder URL"])
        return tfhub_model_map, models_df
    else:
        return tfhub_model_map


def download_and_extract_zip(zip_url, destination_dir):
    """
    Downloads and extracts a zip file to the specified destination.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    zip_filename = os.path.basename(zip_url)

    # Download the zip file
    if not os.path.exists(os.path.join(destination_dir, zip_filename)):
        import wget
        wget.download(zip_url, destination_dir)

    # Extract the zip file
    dataset_zip_path = os.path.join(destination_dir, zip_filename)
    with ZipFile(dataset_zip_path, "r") as zipfile:
        zipfile.extractall(path=destination_dir)
