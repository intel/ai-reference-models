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

# System modules
import argparse
import copy
import json
import os
import sys

# Custom modules
import js_merge

def to_dict(keys, value):
    if len(keys) > 1:
        return { keys[0]: to_dict(keys[1:], value) }
    return { keys[0]: value }

def pairs_to_dict(pairs: list):
    res = {}
    for item in pairs:
        if item == '':
            continue
        pair = item.split('=')
        if len(pair) != 2:
            raise ValueError

        item = to_dict(pair[0].split('.'), pair[1])
        res = js_merge.merge(res, item)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='save_to_json',
          description='Save data to JSON file',
          epilog='Copyright (c) 2024 Intel Corporation')

    parser.add_argument('--amend', action='store_true', help='amend data to output file')
    parser.add_argument('--indent', default=None, help='indent for json.dump()')
    parser.add_argument('file', help='JSON file to store output')
    parser.add_argument('keys', metavar='key=value',  help='key/value pair to amend (key.key=value for nested keys)', nargs='+')

    args = parser.parse_args()

    try:
        to_add = pairs_to_dict(args.keys)
    except Exception as e:
        print('error: invalid key/value pair(s)', file=sys.stderr)
        sys.exit(1)

    data = {}
    if args.amend:
        with open(args.file, 'r') as f:
            data = json.load(f)

    # merging into data giving preference to processed keys and values
    # note: order of arguments and preference to take conflicting values from to_add
    # is important to preserve original data fields order and overrite conflicting
    # values from to_add
    data = js_merge.merge(data, to_add, False)

    indent=int(args.indent) if args.indent and args.indent.isdecimal() else args.indent

    with open(args.file, 'w') as f:
        json.dump(data, f, indent=indent)
