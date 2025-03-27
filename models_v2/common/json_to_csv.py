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
import csv


def base_formatter(value: str):
    return str(value).lower().replace("_", "-").replace(" ", "-")


# Serializes dictionary to have no nested values, i.e.
#   { a: {b: c} }
# will become:
#   { a.b: c }
def serialize(what: dict, path=[], sep: str = ".", format=base_formatter):
    res = {}

    def recurse(res, what, key):
        if isinstance(what[key], dict) or isinstance(what[key], list):
            res |= serialize(what[key], path=path + [key], sep=sep, format=format)
        else:
            key2 = sep.join(list(map(format, path + [key])))
            res[key2] = what[key]

        return res

    if isinstance(what, list):
        for index in range(len(what)):
            res = recurse(res, what, index)
    elif isinstance(what, dict):
        for key in what:
            res = recurse(res, what, key)
    return res


def read_data(files: list):
    data = []
    for file in files:
        with open(file, "r") as f:
            data += [json.load(f)]

    return data


def make_csv(data: list, sep: str = ".", format=base_formatter):
    table = []
    all_keys = set()
    for d in data:
        # Serialize item so it has no nested values
        table += [serialize(d, sep=sep, format=format)]

        # Track all keys
        all_keys |= set(table[-1].keys())

    # Augment serialized data so it has empty values for any keys its missing
    for key in sorted(all_keys, key=lambda x: x):
        for d in table:
            if key not in d:
                d[key] = ""

    return table


def write_table(file: str, csv_table: list):
    with open(file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=csv_table[0].keys())
        writer.writeheader()
        for row in csv_table:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="js_to_csv",
        description="Dump few JSON files content into a single CSV",
        epilog="Copyright (c) 2024 Intel Corporation",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        required=True,
        help="File to store output",
    )
    parser.add_argument("file", help="JSON file to dump into CSV", nargs="+")

    args = parser.parse_args()

    data = read_data(args.file)
    csv_table = make_csv(data)
    write_table(args.output, csv_table)
