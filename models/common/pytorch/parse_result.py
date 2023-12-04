#
# Copyright (c) 2023 Intel Corporation
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

#!/urs/bin/env python
# coding: utf-8


"""
Models log parser.
"""


import argparse
import importlib
import re
from typing import Optional

from common.utils import check_python_version


def get_functional_status(log_path: str, pattern_dict: dict) -> str:
    """Check if log contains functional line."""
    with open(log_path, "r", encoding="UTF-8") as file:
        for line in file:
            match = re.search(pattern_dict["pattern"], line)
            if match:
                return "pass"
    return "fail"


def get_error(log_path: str) -> str:
    """Get error message from log."""
    with open(log_path, "r", encoding="UTF-8") as file:
        for line in file:
            match = re.search(r"\S+Error", line)
            if match:
                return match.group(0)
    return "?"


def parse_log_total(log_path: str, pattern_dict: dict) -> Optional[float]:
    """Parse log and return matched value as total."""
    value = None
    with open(log_path, "r", encoding="UTF-8") as file:
        for line in file:
            match = re.search(pattern_dict["pattern"], line)
            if match:
                value = float(match.group(1))
    return value

def parse_log_total_for_hvd_ddp(log_path: str, pattern_dict: dict) -> Optional[float]:
    """Parse log and return matched value as total."""
    sum_value = []
    with open(log_path, "r", encoding="UTF-8") as file:
        for line in file:
            match = re.search(pattern_dict["pattern"], line)
            if match:
                sum_value.append(float(match.group(1)))
    if sum_value:
        return float(sum(sum_value[-len(sum_value):]))
    return None

def parse_log_min_for_hvd_ddp(log_path: str, pattern_dict: dict) -> Optional[float]:
    """Parse log and return matched value as total."""
    min_value = []
    with open(log_path, "r", encoding="UTF-8") as file:
        for line in file:
            match = re.search(pattern_dict["pattern"], line)
            if match:
                min_value.append(float(match.group(1)))
    if min_value:
        return float(min(min_value))
    return None

def parse_log_mean(log_path: str, pattern_dict: dict) -> Optional[float]:
    """Parse log and return mean of matched values."""
    partials = []
    with open(log_path, "r", encoding="UTF-8") as file:
        for line in file:
            match = re.search(pattern_dict["pattern"], line)
            if match:
                partials.append(float(match.group(1)))
    number_of_partials = pattern_dict.get("number_of_partials", len(partials))
    if partials:
        return float(sum(partials[-number_of_partials:]) / number_of_partials)
    return None


def parse_log(test_type: str,
              log_dir: str,
              batch_size: str) -> None:
    """Parse log and save results."""
    test_case = log_dir.split("/")[-1].replace('_raw.log', '')
    model = test_case.split("_")[0].strip()
    mode = test_case.split("_")[2].split("-")[0].strip()
    patterns = importlib.import_module(f"consts.patterns.{test_type}.{mode}.{model}")
    acc_pattern = patterns.ACC
    perf_pattern = patterns.PERF
    functional_pattern = patterns.FUNCTIONAL

    # parse accuracy
    accuracy = "N/A"
    accuracy_unit = ""
    if acc_pattern.get("unit"):
        accuracy_unit = acc_pattern['unit']
        if test_type == "hvd" or test_type == "ddp":
            if acc_pattern.get("type") == "total":
                accuracy = parse_log_total_for_hvd_ddp(log_dir, acc_pattern)
            elif acc_pattern.get("type") == "mean":
                accuracy = parse_log_mean(log_dir, acc_pattern)
        else:
            if acc_pattern.get("type") == "total":
                accuracy = parse_log_total(log_dir, acc_pattern)
            elif acc_pattern.get("type") == "mean":
                accuracy = parse_log_mean(log_dir, acc_pattern)

    # parse performance
    performance = "N/A"
    performance_unit = ""
    if perf_pattern.get("unit"):
        performance_unit = perf_pattern["unit"]
        if test_type == "hvd" or test_type == "ddp":
            performance_min = parse_log_min_for_hvd_ddp(log_dir, perf_pattern)
            if perf_pattern["type"] == "total":
                performance = parse_log_total_for_hvd_ddp(log_dir, perf_pattern)
            elif perf_pattern["type"] == "mean":
                performance = parse_log_mean(log_dir, perf_pattern)
        else:
            if perf_pattern["type"] == "total":
                performance = parse_log_total(log_dir, perf_pattern)
            elif perf_pattern["type"] == "mean":
                performance = parse_log_mean(log_dir, perf_pattern)
        # inverse if needed
        if perf_pattern.get("inverse") and performance:
            performance = 1 / performance
        # multiply if needed
        if perf_pattern.get("multiply") and performance:
            multiply = perf_pattern["multiply"]
            performance = performance * multiply
        # use batch size if needed
        if perf_pattern.get("use_batch_size") and performance:
            performance = performance * int(batch_size)

    # get latency
    latency = "N/A"
    if test_type == "hvd" or test_type == "ddp":
        latency = "N/A"
    else:
        if perf_pattern.get("unit") and isinstance(performance, float):
            performance_unit = perf_pattern["unit"]
            if bool(re.search('/s$', performance_unit)) or performance_unit == "fps":
                latency = int(batch_size) / performance

    # get functional status
    functional_status = get_functional_status(log_dir, functional_pattern)

    performance = f"{float(f'{performance:.8g}')}" if isinstance(performance, float) else performance
    if test_type == "hvd" or test_type == "ddp":
        performance_min = f"{float(f'{performance_min:.8g}')}" if isinstance(performance_min, float) else performance_min
    accuracy = f"{accuracy:.3f}" if isinstance(accuracy, float) else accuracy

    # save results to log file
    result_dir = log_dir.replace('_raw.log', '.log')
    result_file = open(result_dir, 'w')
    result_file.write(f"Batch Size: {batch_size}")
    if test_type == "hvd" or test_type == "ddp":
        result_file.write(f"\nSum Performance: {performance} {performance_unit}")
        result_file.write(f"\nMin Performance: {performance_min} {performance_unit}")
    else:
        result_file.write(f"\nPerformance: {performance} {performance_unit}")
    result_file.write(f"\nLatency: {latency} s")
    result_file.write(f"\nAccuracy: {accuracy_unit} {accuracy}")
    result_file.write(f"\nFunctional: {functional_status}")
    if functional_status == "fail":
        error = get_error(log_dir)
        result_file.write(f"\nError: {error}")
    result_file.write(f"\n")
    result_file.close()


def main():
    """
    Main function of models results parser.
    """
    check_python_version()
    parser = argparse.ArgumentParser(description=f"Model result parser")
    parser.add_argument('--test_type', '-t', default="anl", choices=["anl", "ddp", "hvd", "por"],
                        help="Test scope type")
    parser.add_argument('--batch_size', '-b', required=True, help='Batch size')
    parser.add_argument('--log_dir', '-l', required=True, help='Log directory')
    args = parser.parse_args()
    parse_log(args.test_type, args.log_dir, args.batch_size)


if __name__ == "__main__":
    main()
