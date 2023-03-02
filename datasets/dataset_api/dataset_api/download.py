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

#

import os
import sys
import requests
import json
from tqdm import tqdm


def download_dataset(dataset_name, dataset_directory):
    # Load the JSON file that contains the dataset URLs
    with open('datasets_urls.json', 'r') as f:
        dataset_urls = json.load(f)

    # Check if the dataset name is in the JSON file
    if dataset_name not in dataset_urls:
        print(f'{dataset_name} is not a valid dataset name')
        return
    # Check if the dataset destination directory is given or create one using the dataset name
    if not dataset_directory:
        dataset_directory = os.path.join(os.getcwd(), dataset_name)
    if not os.path.isdir(dataset_directory):
        os.makedirs(dataset_directory)

    # Get the URL for the dataset
    dataset_url = dataset_urls[dataset_name]
    for url in dataset_url['urls']:
        response = requests.get(url['url'])

        if response.status_code == 200:
            # Check if the key 'file_name' exists in the JSON file
            if 'file_name' in url:
                filename = url['file_name']
            else:
                filename = url['url'].split('/')[-1]

            destination_file_path = os.path.join(dataset_directory, filename)
            if os.path.exists(destination_file_path):
                print("\nFile already exists in {}.".format(destination_file_path))
                print("Please delete it and try again!.\n")
            else:
                # Create a progress bar object and iterate over the range
                my_range = range(100)
                for i in tqdm(my_range, desc="Downloading", unit="item"):
                    # Download the file if it does not exist in the desired dataset directory
                    with open(destination_file_path, 'wb') as f:
                        f.write(response.content)
                print(f"{filename} downloaded successfully in {dataset_directory}")
        else:
            sys.exit("\nError while downloading {}. Status code: {}".format(filename, response.status_code))
