#!/usr/bin/env bash
#
# Copyright (c) 2020 Intel Corporation
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

PATTERN='[-a-zA-Z0-9_]*='
if [ $# -eq 0 ] || [ $# -gt 4 ]; then
    echo 'ERROR:'
    echo "Expected 1-4 parameters got $#"
    printf 'Please use following parameters:
    --model=model to be inferenced
    --batch_size=number of samples per batch (default 1)
    --servers="server1:port;server2:port" (default "localhost:8500")
    --clients=number of clients per server (default 1)
    '
    exit 1
fi

batch_size=1
clients=1
servers="localhost:8500"

for i in "$@"
do
    case $i in
        --model=*)
            model=`echo $i | sed "s/${PATTERN}//"`;;
	--batch_size=*)
            batch_size=`echo $i | sed "s/${PATTERN}//"`;;
        --servers=*)
            servers=`echo $i | sed "s/${PATTERN}//"`;;
        --clients=*)
            clients=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

if [ -v ${model} ]; then
    echo "Parameter model is required!";
    exit 1
fi

server_list=($(echo "$servers" | tr ';' '\n'))

for client in `seq ${clients}`
do
    for server in "${server_list[@]}"
    do 
        command="python run_tf_serving_client.py --model ${model} --batch_size ${batch_size} --server ${server}";
	${command} &
    done
done
