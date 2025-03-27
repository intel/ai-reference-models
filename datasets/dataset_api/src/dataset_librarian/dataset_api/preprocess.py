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

import json
import os
import sys
import signal
import subprocess
import pkg_resources


def preprocess_dataset(dataset_name, dataset_directory, split_ratio):
    # Get the path to the file relative to the package root
    datasets = pkg_resources.resource_filename(
        "dataset_librarian", "datasets_urls.json"
    )
    # Load the JSON file that contains the dataset URLs
    with open(datasets, "r") as f:
        dataset_urls = json.load(f)

    # Check if the dataset name is in the JSON file
    if dataset_name not in dataset_urls:
        sys.exit("\nError: {} is not a valid dataset name.\n".format(dataset_name))

    if dataset_directory:
        os.environ["DATASET_DIR"] = dataset_directory
    else:
        sys.exit(
            "\nError: Please set --directory or -d to the location of the raw dataset directory.\n"
        )

    if not os.path.isdir(dataset_directory):
        print("\nError: {} does not exist!".format(dataset_directory))
        sys.exit(
            """Please specifiy the path to the raw dataset directory using --directory flag.\n"""
            """Or use --download flag to download the raw dataset prior to preprocessing.\n"""
        )

    datasets_api_path = os.path.dirname(os.path.realpath(__file__))
    scripts_path = os.path.join(datasets_api_path, os.pardir, "scripts")

    if dataset_name == "brca":
        brca_scripts = os.path.join(scripts_path, dataset_name)
        nlp_data_script = os.path.join(brca_scripts, "prepare_nlp_data.py")
        vision_data_script = os.path.join(brca_scripts, "prepare_vision_data.py")
        nlp_split_data = os.path.join(brca_scripts, "create_data_split.py")
        vision_split_data = os.path.join(brca_scripts, "create_data_split.py")
        # Run dataset preprocessing scripts
        _launch_command(["python", nlp_data_script])
        _launch_command(["python", vision_data_script])

        # Split the pre-processed data:
        _launch_command(["python", nlp_split_data, "--split_ratio", str(split_ratio)])
        _launch_command(
            ["python", vision_split_data, "--split_ratio", str(split_ratio)]
        )

    elif dataset_name == "mvtec-ad":
        raw_datset = os.path.join(dataset_directory, "mvtec_anomaly_detection.tar.xz")
        if os.path.exists(raw_datset):
            # extract dataset files
            subprocess.run(["tar", "-xf", raw_datset, "--directory", dataset_directory])
        elif not os.listdir(dataset_directory):
            print("\nError: No datasets found in {}.\n".format(dataset_directory))
            sys.exit(
                "\nPlease use --download flag to download the raw dataset prior to preprocessing.\n"
            )

        mvtec_scripts = os.path.join(scripts_path, dataset_name)
        csv_generator_script = (
            "python "
            + os.path.join(mvtec_scripts, "csv_generator_mvtec.py")
            + " --path "
            + dataset_directory
        )
        # Run dataset preprocessing scripts
        os.system(csv_generator_script)
    else:
        sys.exit(
            "\nError: Preprocessing is not currently supported for {} dataset.\n".format(
                dataset_name
            )
        )


def _launch_command(run_cmd):
    """runs command that runs the dataset script on bare metal and exits on ctrl c"""
    p = subprocess.Popen(run_cmd, preexec_fn=os.setsid)
    try:
        p.communicate()
    except KeyboardInterrupt:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
