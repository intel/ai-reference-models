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
import os
import math
import json
import sys

columns_of_interest = {}

# Define list of items we want to log from xpu-smi output for Flex/Max XPU
columns_of_interest["Flex"] = [
    "GPU Utilization (%)",
    "GPowerPU  (W)",
    "GPU Frequency (MHz)",
    "GPU Core Temperature (Celsius Degree)",
    "GPU EU Array Active (%)",
    "GPU EU Array Stall (%)",
    "GPU EU Array Idle (%)",
    "GPU Memory Used (MiB)",
    "Compute Engine 0 (%)",
    "Compute Engine 1 (%)",
    "Compute Engine 2 (%)",
    "Compute Engine 3 (%)",
    "Decoder Engine 0 (%)",
    "Decoder Engine 1 (%)",
    "Copy Engine 0 (%)",
    "Media Enhancement Engine 0 (%)",
    "Media Enhancement Engine 1 (%)",
    "Media Engine Frequency (MHz)",
    "PCIe Read (kB/s)",
    "PCIe Write (kB/s)",
]
columns_of_interest["Max"] = columns_of_interest["Flex"]

# Define list of items we want to log from nvidia-smi output for CUDA GPU
columns_of_interest["CUDA"] = [
    "fan.speed [%]",
    "temperature.gpu",
    "utilization.gpu [%]",
    "utilization.memory [%]",
    "memory.total [MiB]",
    "memory.free [MiB]",
    "memory.used [MiB]",
    "encoder.stats.sessionCount",
    "encoder.stats.averageFps",
    "encoder.stats.averageLatency",
    "power.draw [W]",
    "power.limit [W]",
    "enforced.power.limit [W]",
    "power.default_limit [W]",
    "power.min_limit [W]",
    "power.max_limit [W]",
    "clocks.current.graphics [MHz]",
    "clocks.current.sm [MHz]",
    "clocks.current.memory [MHz]",
    "clocks.current.video [MHz]",
    "clocks.max.graphics [MHz]",
    "clocks.max.sm [MHz]",
    "clocks.max.memory [MHz]",
]

platform_choices = ["Flex", "Max", "CUDA"]


def process_csv_row(row):
    row = row.strip()
    while ", " in row:
        row = row.replace(", ", ",")
    while "  " in row:
        row = row.replace("  ", " ")
    return list(map(lambda x: x.strip(), row.split(",")))


def reformat_key(key):
    key = key.replace("[", "(")
    key = key.replace("]", ")")
    key = key.replace(" ", "-")
    key = key.replace(".", "-")
    key = key.split("-(")

    base_key = None
    unit = None
    if len(key) == 2:
        base_key = key[0].lower()
        unit = key[1][:-1]
    else:
        base_key = key[0].lower()
        unit = ""

    return base_key, unit


def read_output(file):
    try:
        with open(file, "r") as file_object:
            results = json.load(file_object)
    except Exception as e:
        print("error: " + str(e), file=sys.stderr)
        sys.exit(1)

    return results


def open_smi_data(path, platform):
    smi_file_name = {
        "Flex": "xpu_smi_dump.csv",
        "Max": "xpu_smi_dump.csv",
        "CUDA": "nvidia_smi_dump.csv",
    }[platform]

    input_file = os.path.join(path, smi_file_name)
    try:
        smi_dump = open(input_file, "r")
    except Exception as e:
        print("error: " + str(e), file=sys.stderr)
        sys.exit(1)

    return smi_dump


def process(smi_dump, platform, results=None):
    if results == None:
        results = {}

    # Read first line from SMI CSV to get header
    header = process_csv_row(smi_dump.readline())

    # Read all samples from SMI CSV
    data = {}
    for row in smi_dump:
        row = process_csv_row(row)
        if len(row) != len(header):
            continue  # skip any corrupted lines
        for column in range(len(row)):
            item = row[column]
            key = header[column]
            if key in columns_of_interest[platform]:
                # Some rows do not have valid data. Skip these.
                if item.lower() in ["n/a", "na", ""]:
                    continue

                # Log the sample as a float
                if key not in data:
                    data[key] = []
                item = item.split()[0]
                try:
                    data[key] += [float(item)]
                except:
                    pass

    # Calculate statistics for various metrics captured in SMI CSV
    if "results" not in results:
        results["results"] = {}
    if "metrics" not in results["results"]:
        results["results"]["metrics"] = {}
    metrics = results["results"]["metrics"]
    for key in data:
        if len(data[key]) > 0:
            formatted_key, unit = reformat_key(key)
            metrics[formatted_key] = {}
            metrics[formatted_key]["min"] = min(data[key])
            metrics[formatted_key]["max"] = max(data[key])
            metrics[formatted_key]["avg"] = sum(data[key]) / len(data[key])
            metrics[formatted_key]["stdev"] = math.sqrt(
                sum([(item - metrics[formatted_key]["avg"]) ** 2 for item in data[key]])
                / len(data[key])
            )
            metrics[formatted_key]["units"] = unit

    return results


def get_parser():
    parser = argparse.ArgumentParser(
        prog="parse_telemetry",
        description="Append summarized xpu-smi or nvidia-smi output into a JSON file",
        epilog="Copyright (c) 2024 Intel Corporation",
    )

    parser.add_argument(
        "--input", default=None, help="path to read CSV output from SMI tool"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON file to write (or append to if already exists) SMI data to",
    )
    parser.add_argument(
        "--platform",
        help="System on which telemetry is being collected",
        default="Flex",
        choices=platform_choices,
    )


def main():
    args = get_parser().parse_args()

    # Open source SMI CSV and load results.json if it exists
    results = read_output(args.output)

    # Determine name for SMI data and open it
    smi_dump = open_smi_data(args.input, args.platform)

    # Process the SMI data
    results = process(smi_dump, args.platform, results)

    # Close SMI CSV
    smi_dump.close()

    # Write updated JSON out
    try:
        with open(args.output, "w") as file_object:
            json.dump(results, file_object, indent=4)
    except Exception as e:
        print("error: " + str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
