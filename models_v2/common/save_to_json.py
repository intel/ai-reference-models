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
import json
import os
import sys
import re

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

def add_to_file(file, to_add, amend=False, indent=0):
    data = {}
    if amend:
        with open(file, 'r') as f:
            data = json.load(f)

    # merging into data giving preference to processed keys and values
    # note: order of arguments and preference to take conflicting values from to_add
    # is important to preserve original data fields order and overrite conflicting
    # values from to_add
    data = js_merge.merge(data, to_add, False)

    with open(file, 'w') as f:
        json.dump(data, f, indent=indent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='save_to_json',
          description='Save data to JSON file',
          epilog='Copyright (c) 2024 Intel Corporation')

    parser.add_argument('--directory', action='store_true', help='if specified, target argument points to a directory within which key-values pairs will be saved to all matched json files')
    parser.add_argument('--recursive', action='store_true', help='if specified and operating in directory mode, key-values pairs will be saved to all matched json files recursively')
    parser.add_argument('--pattern', default=None, help='if operating in directory mode, only json files that match this regex expression will be modified')
    parser.add_argument('--amend', action='store_true', help='amend data to output file')
    parser.add_argument('--indent', default=None, help='indent for json.dump()')
    parser.add_argument('target', help='JSON file (or directory if in directory mode) to store output')
    parser.add_argument('keys', metavar='key=value',  help='key/value pair to amend (key.key=value for nested keys)', nargs='+')

    args = parser.parse_args()

    pattern = None
    if args.pattern:
        pattern = re.compile('{}'.format(args.pattern))

    complete_file_list = []
    if not os.path.exists(args.target):
        print('error: no such file or directory: ' + args.target, file=sys.stderr)
        sys.exit(1)
    if args.directory:
        if not os.path.isdir(args.target):
            print('error: not a directory: ' + args.target, file=sys.stderr)
            sys.exit(1)
        for root, dirs, files in os.walk(args.target):
            for file in files:
                if not file.lower().endswith('.json'):
                    continue
                if args.pattern:
                    match_result = re.search(pattern, file)
                    if not match_result:
                        continue
                complete_file_list += [os.path.join(root, file)]
            if not args.recursive:
                # If we are not operating recursively, then break out of loop after processing top level of specified directory
                break
    else:
        if not os.path.isfile(args.target):
            print('error: not a file: ', args.target, file=sys.stderr)
            sys.exit(1)
        complete_file_list += [args.target]

    try:
        to_add = pairs_to_dict(args.keys)
    except Exception as e:
        print('error: invalid key/value pair(s)', file=sys.stderr)
        sys.exit(1)

    indent=int(args.indent) if args.indent and args.indent.isdecimal() else args.indent

    for file in complete_file_list:
        add_to_file(file, to_add, args.amend, indent=indent)

    
