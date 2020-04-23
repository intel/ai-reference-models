# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Download and clean the Census Income Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import requests

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/tmp/census_data',
                    help='Directory to download census data')
parser.add_argument('--http_proxy', type=str, default=None)
parser.add_argument('--https_proxy', type=str, default=None)


def download_and_clean_file(filename, url):
    """Downloads data from url, and makes changes to match the CSV format."""
    proxies = {}
    print(filename)
    if ARGS.http_proxy:
        proxies['http'] = ARGS.http_proxy
    if ARGS.https_proxy:
        proxies['https'] = ARGS.https_proxy
    try:
        request = requests.get(url, stream=True, proxies=proxies)
        request.raise_for_status()
        with open(filename, 'wb') as eval_file:
            for line in request.iter_lines():
                line = line.strip()
                line = line.replace(b', ', b',')
                if not line or b',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
                line += b'\n'
                eval_file.write(line)
    except requests.exceptions.HTTPError as err:
        print(err)
        sys.exit(1)


def main():
    if not os.path.exists(ARGS.data_dir):
        os.mkdir(ARGS.data_dir)

    training_file_path = os.path.join(ARGS.data_dir, TRAINING_FILE)
    download_and_clean_file(training_file_path, TRAINING_URL)

    eval_file_path = os.path.join(ARGS.data_dir, EVAL_FILE)
    download_and_clean_file(eval_file_path, EVAL_URL)


if __name__ == '__main__':
    ARGS = parser.parse_args()
    main()
    print("Wide & Deep dataset is downloaded at {}".format(ARGS.data_dir))
