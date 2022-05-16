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

import os

from zipfile import ZipFile


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
