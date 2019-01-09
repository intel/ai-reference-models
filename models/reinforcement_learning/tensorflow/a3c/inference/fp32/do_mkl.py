#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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
# SPDX-License-Identifier: EPL-2.0
#

import sys
import operator
state = 0
typeNames = {}
result = {}
result1 = {}
fn = sys.argv[1]
ss = int(sys.argv[2]) # step start to measure
with open(fn) as fp:
    for line in fp:
        if state < ss and "iter" in line:
            state = state + 1
        elif state == ss and "time: " in line:
            start = line.find('time: ') + 6
            end = line.find(' ms', start)
            value = line[start:end]
            start = end + 4
            end = line.find('<', start)
            name = line[start:end]
            if result.has_key(name):
                result[name] += float(value)
            else:
                result[name] = float(value)
        elif state == ss and "iter" in line:
            state = state + 1
        elif state == ss + 1 and "time: " in line:
            start = line.find('time: ') + 6
            end = line.find(' ms', start)
            value = line[start:end]
            start = end + 4
            end = line.find('<', start)
            name = line[start:end]
            if result1.has_key(name):
                result1[name] += float(value)
            else:
                result1[name] = float(value)
        elif state == ss + 1 and "iter" in line:
            state = state + 1
            break
for k, v in result.items():
    if result1.has_key(k):
        result[k] = (v + result1[k]) / 2
result = sorted(result.items(), key=operator.itemgetter(1), reverse = True)
for t in result:
    print("%s, %f" % t )
