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
import sys


def merge(a: dict, b: dict, prefer_a: bool = True, path=[]):
    c = copy.deepcopy(a)
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                c[key] = merge(a[key], b[key], prefer_a, path + [str(key)])
            elif type(a[key]) != type(b[key]):
                print(
                    "error: type conflict for the key value: "
                    + ".".join(path + [str(key)])
                )
                sys.exit(1)
            elif a[key] != b[key]:
                print(
                    "info: different values for the key:" + ".".join(path + [str(key)]),
                    file=sys.stderr,
                )
                if not prefer_a:
                    print("info: using value from dict-b: " + str(b[key]))
                    c[key] = b[key]
                else:
                    print("info: using value from dict-a: " + str(a[key]))
        else:
            c[key] = b[key]
    return c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="js_merge",
        description="Merge few JSON files together",
        epilog="Copyright (c) 2024 Intel Corporation",
    )

    parser.add_argument("--indent", default=None, help="indent for json.dump()")
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        default="",
        help="File to store output",
    )
    parser.add_argument("file", help="JSON file to merge", nargs="+")

    args = parser.parse_args()

    data = {}
    for file in args.file:
        with open(file, "r") as f:
            data = merge(data, json.load(f))

    indent = (
        int(args.indent) if args.indent and args.indent.isdecimal() else args.indent
    )
    if args.output == "":
        json.dump(data, sys.stdout, indent=indent)
    else:
        with open(args.output, "w") as f:
            json.dump(data, f, indent=indent)
