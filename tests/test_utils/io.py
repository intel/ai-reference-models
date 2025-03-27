#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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
import json


def parse_json_files(json_dir_path):
    """
    Reads the JSON files in the specified directory.  Checks for a value number of columns in
    each row. Returns the JSON files values as a list of tuples.
    """
    values = []
    for model_file in os.listdir(json_dir_path):
        file_path = os.path.join(json_dir_path, model_file)
        with open(file_path) as f:
            data = json.load(f)
            for x in data:
                # Use 0-111 as the default cpuset, if it's not specified in the json
                cpuset = "0-111"
                if "cpuset" in x.keys():
                    cpuset = x["cpuset"]
                values.append(
                    tuple(
                        (
                            x["input"],
                            x["output"],
                            model_file + " :: " + x["_comment"],
                            cpuset,
                        )
                    )
                )
    return values
