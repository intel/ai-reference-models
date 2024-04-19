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
import sys
import json
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_dataset

set_id='cnn_dailymail'
version='3.0.0'
instruction_template="Summarize the following news article:"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--split", default="validation", help="Split to use")
    parser.add_argument("--output-dir", help="Output directory")

    return parser.parse_args()

def check_path(path):
    return os.path.exists(path)

def prepare_calibration_data(split, output_dir):

    dataset = load_dataset("cnn_dailymail", name="3.0.0", split=split)
    train = dict((x['id'], x) for x in dataset)
    
    inputs = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        x = dict()
        x["instruction"] = instruction_template
        x["input"] = sample["article"]
        x["output"] = sample["highlights"]
        inputs.append(x)
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir,"cnn_dailymail_{}.json".format(split))
    with open(output_path, 'w') as write_f:
        json.dump(inputs, write_f, indent=4, ensure_ascii=False)

    print("{} data saved at {}".format(split, output_path))

def main():

    args = get_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "cnn-dailymail-{}".format(args.split)
    prepare_calibration_data(args.split, output_dir)

if __name__=="__main__":
    main()

