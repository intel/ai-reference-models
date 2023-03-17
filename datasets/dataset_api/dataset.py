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

os.system("source ~/.bashrc")
user_input = os.environ.get("USER_CONSENT")

if user_input == "y":
    import json
    import argparse
    from dataset_api.download import download_dataset
    from dataset_api.preprocess import preprocess_dataset

    # Create the parser for the CLI
    parser = argparse.ArgumentParser(description='Download and preprocess datasets')
    parser.add_argument('-n', '--name', type=str, help='name of the dataset to download')
    parser.add_argument('-l', '--list', action='store_true', help='list the supported datasets')
    parser.add_argument('-d', '--directory', type=str, help='the desired dataset directory location')
    parser.add_argument('--download', action='store_true', help='download the raw dataset')
    parser.add_argument('--preprocess', action='store_true', help='preprocess the dataset')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Download the dataset if the --download flag is true
    if args.download:
        download_dataset(args.name, args.directory)

    # Preprocess the dataset if the --preprocess flag is true
    if args.preprocess:
        preprocess_dataset(args.name, args.directory)

    # List the supported datasets if the --list flag is true
    if args.list:
        with open('datasets_urls.json') as f:
            datasets = json.load(f)
        # Get the list of keys
        keys = list(datasets.keys())
        # Print the list of keys
        print(keys)

else:
    print("Please read and accept terms and conditions to be able to to use the dataset API.")
    print("\nRun ./setup.sh to view and accept the terms and conditions.\n")
