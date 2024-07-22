# Copyright (c) 2023-2024 Intel Corporation
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

# system modules
import io
import os
import json
import sys

# sample modules
import ifrnet_util

def stdout_helper(output_str):
    sys.stdout.write(output_str + '\n')
    sys.stdout.flush()

def stderr_helper(output_str):
    sys.stderr.write(output_str + '\n')
    sys.stderr.flush()

def get_prefix(prefix):
    if 'instance' in vars(ifrnet_util.args):
        return '{0}[{1}/{2}]'.format(prefix, ifrnet_util.args.instance, ifrnet_util.args.streams)
    else:
        return '{0}'.format(prefix)

def write_info(info_str):
    output_str = '{0}: {1}'.format(get_prefix('INFO'), info_str)
    stdout_helper(output_str)

def write_warning(warning_str):
    output_str = '{0}: {1}'.format(get_prefix('WARNING'), warning_str)
    stdout_helper(output_str)

def write_error(error_str):
    output_str = '{0}: {1}'.format(get_prefix('ERROR'), error_str)
    stderr_helper(output_str)

def read_json(target, error_on_fail=False):
    if not os.path.exists(target):
        if error_on_fail:
            write_error('No such JSON file: {0}'.format(target))
            sys.exit(1)
        else:
            return json.loads('{}')
    file_object = io.open(target, 'r', encoding='utf-8')
    json_object = json.load(file_object)
    file_object.close()
    return json_object

def write_json(target, data):
    file_object = io.open(target, 'w', encoding='utf-8')
    json.dump(data, file_object, indent=4)
    file_object.close()
