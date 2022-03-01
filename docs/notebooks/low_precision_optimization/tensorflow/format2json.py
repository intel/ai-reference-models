#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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


def transfer_res_to_json(prefix):
    print("Performance for "+prefix)
    with open(prefix + "_throughput.txt") as f:
        throughput = float(f.readlines()[0].split(" ")[1])
        print(throughput)

    with open(prefix + "_latency.txt") as f:
        latency = float(f.readlines()[0].split(" ")[2])
        print(latency)

    with open(prefix + "_accuracy.txt") as f:
        lines = f.readlines()
        accus = []
        for line in lines:
            accu = float(line.split(" ")[-2][1:-1])
            accus.append(accu)
        accuracy = sum(accus)/len(accus)        
        print(accuracy)
        
    json_file = prefix+".json"
    with open(json_file, "w") as f:
        res={'accuracy':accuracy, 'throughput':throughput, 'latency':latency}
        json.dump(res, f)
        print("Save to "+json_file+"\n")

transfer_res_to_json("fp32")
transfer_res_to_json("int8")
