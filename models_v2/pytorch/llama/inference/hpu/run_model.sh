#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Intel Corporation
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

#!/bin/bash

 
# bypass all benchmark for log
cd /workspace/optimum-habana/examples/text-generation/
python3  Benchmark.py > oh_benchmark.log

# docker cp out the log
summary_log=oh_benchmark.log
throughput=$(cat $summary_log | grep Gaudi.json)

yaml_content=$(cat <<EOF
results:
 - key: throughput
   value: $throughput
   unit: it/s
EOF
)

# Write the content to a YAML file
echo "$yaml_content" >  ../results.yaml

