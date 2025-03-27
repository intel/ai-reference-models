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

import os
from dotenv import load_dotenv, dotenv_values, set_key
import pkg_resources
import json
import argparse
from dataset_librarian.dataset_api.download import download_dataset
from dataset_librarian.dataset_api.preprocess import preprocess_dataset


def accept_terms_and_conditions(package_path):
    print(
        "Please read and accept terms and conditions to be able to to use the dataset API."
    )
    print(
        "*********************************************************************************"
    )
    with open(package_path + "/terms_and_conditions.txt", "r") as f:
        print(f.read())
    print(
        "*********************************************************************************"
    )
    while True:
        acceptance = input("Do you agree to the terms and conditions? (y/n): ")
        acceptance = acceptance.lower()
        if acceptance == "n":
            print("Terms and conditions disagreed")
            break
        elif acceptance == "y":
            print("Terms and conditions agreed")
            break
        else:
            print("Invalid Response. Please choose: y or n")
    return acceptance


def main():
    package_name = "dataset_librarian"
    package_path = pkg_resources.get_distribution(package_name).location
    package_path = os.path.join(package_path, package_name)
    # Check if the .env file exists
    env_file_path = os.path.join(package_path, ".env")
    if not os.path.isfile(env_file_path):
        with open(env_file_path, "w") as file:
            file.write("USER_CONSENT=None\n")
    # Extract key
    load_dotenv()
    USER_CONSENT = dotenv_values(env_file_path).get("USER_CONSENT")
    # Present terms and conditions if needed
    if USER_CONSENT != "y":
        acceptance = accept_terms_and_conditions(package_path)
        set_key(env_file_path, "USER_CONSENT", acceptance)
    USER_CONSENT = dotenv_values(env_file_path).get("USER_CONSENT")
    # Validate terms and conditions
    if USER_CONSENT != "y":
        quit()

    # Create the parser for the CLI
    parser = argparse.ArgumentParser(description="Download and preprocess datasets")
    parser.add_argument(
        "-n", "--name", type=str, help="name of the dataset to download"
    )
    parser.add_argument(
        "-l", "--list", action="store_true", help="list the supported datasets"
    )
    parser.add_argument(
        "-d", "--directory", type=str, help="the desired dataset directory location"
    )
    parser.add_argument(
        "--download", action="store_true", help="download the raw dataset"
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="preprocess the dataset"
    )
    parser.add_argument(
        "--split_ratio", type=float, help="split ratio of the test data", default=0.1
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Download the dataset if the --download flag is true
    if args.download:
        download_dataset(args.name, args.directory)

    # Preprocess the dataset if the --preprocess flag is true
    if args.preprocess:
        preprocess_dataset(args.name, args.directory, args.split_ratio)

    # List the supported datasets if the --list flag is true
    if args.list:
        datasets = pkg_resources.resource_filename(
            "dataset_librarian", "datasets_urls.json"
        )
        with open(datasets) as f:
            datasets = json.load(f)
        # Get the list of keys
        keys = list(datasets.keys())
        # Print the list of keys
        print(keys)


if __name__ == "__main__":
    main()
