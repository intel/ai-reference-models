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
import csv
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import json
import multiprocessing as mp

# Custom modules
import js_merge
import js_sysinfo
import json_to_csv
import parse_telemetry
import telemetry
import save_to_json

platform_choices = ["Flex", "Max", "CUDA"]


def get_config(metadata: str, app_args: list):
    res = {
        "config": {
            "metadata": {
                # getting version in a form '2024.Q1'
                "version": "{0}.Q{1}".format(
                    datetime.datetime.now().year,
                    (datetime.datetime.now().month - 1) // 3 + 1,
                )
            },
            "workload": {"cmdline": app_args},
        }
    }
    return js_merge.merge(res, save_to_json.pairs_to_dict(metadata.split(" ")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Benchmark application given with templated cmdline arguments",
        epilog="Copyright (c) 2024 Intel Corporation",
    )

    # parser.add_argument('-v', '--verbose', action="store_true", help='Enable verbose output')
    # parser.add_argument('-q', '--quiet', action="store_true", help='Be quiet and suppress messages to stdout')

    parser.add_argument("--indent", default=None, help="indent for json.dump()")
    parser.add_argument(
        "--metadata",
        action="store",
        type=str,
        default="",
        help="Space separated key=value pairs to amend to json reports",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        default="",
        required=True,
        help="Path to store outputs",
    )
    parser.add_argument(
        "--profile",
        action="store",
        type=str,
        default="",
        required=True,
        help="Profile with tests list",
    )
    parser.add_argument(
        "--platform",
        help="System on which telemetry is being collected",
        default="",
        required=True,
        choices=platform_choices,
    )
    parser.add_argument(
        "--socket",
        action="store",
        type=str,
        default="/tmp/telemetry.s",
        help="Socket to control telemetry capture",
    )
    parser.add_argument(
        "--telemetry", action="store_true", help="enable GPU telemetry capture"
    )

    args, benchmark_app_args = parser.parse_known_args()

    indent = (
        int(args.indent) if args.indent and args.indent.isdecimal() else args.indent
    )

    # getting tests list from given profile
    tests = []
    with open(args.profile, "r") as f:
        ps = csv.DictReader(f)
        for p in ps:
            tests += [p]

    print("Profile: " + args.profile)
    print("Tests: {}".format(len(tests)))

    # configuring output directory
    # ultimately we will have the following layout of output directory:
    #
    # .
    # ├── profile.csv               # copy of profile
    # ├── sysinfo.json              # baremetal sysinfo description
    # ├── test_{i}                  # output directory of i-th test
    # │   ├── test.csv              # test csv definition (i-th line from profile.csv)
    # │   ├── results.json          # results output from the test complying with the schema
    # │   ├── *_smi_dump.csv        # raw SMI output
    # │   ├── telemetry.json        # processed telemetry stats from SMI
    # │   └── *                     # whatever other output files test produces
    # ├── results_test_{i}.json     # ultimate report for i-th test
    # └── summary.csv               # engineering summary for debug
    #
    # Fail if can't create the directory or if it exists.
    pathlib.Path(args.output_dir).mkdir()

    # copy the whole profile
    shutil.copyfile(args.profile, os.path.join(args.output_dir, "profile.csv"))

    # get system description
    sysinfo_parser = js_sysinfo.get_parser()
    sysinfo_args = sysinfo_parser.parse_args(["--all", "--verbose"])
    sysinfo = js_sysinfo.get_sysinfo(**vars(sysinfo_args))

    # dump system description to .json file
    file = os.path.join(args.output_dir, "sysinfo.json")
    with open(file, "w") as f:
        json.dump(sysinfo, f, indent=indent)

    # track results for summary CSV
    to_summarize = []

    # define tracker for set of spawned processes
    processes = {}

    for iteration, test in enumerate(tests):
        print("Running: {0}/{1}".format(iteration + 1, len(tests)))

        # create output directory for the test
        output_dir = os.path.join(args.output_dir, "test_" + str(iteration + 1))
        pathlib.Path(output_dir).mkdir()

        # start telemetry capture in separate process
        if args.telemetry:
            processes["telemetry.capture"] = {
                "process": mp.Process(target=telemetry.capture, args=(args, output_dir))
            }
            processes["telemetry.capture"]["process"].start()

        # dump test csv definition
        with open(os.path.join(output_dir, "test.csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=test.keys())
            writer.writeheader()
            writer.writerow(test)

        # creating benchmark app cmdline from the template
        app_args = list(benchmark_app_args)
        for arg in range(len(app_args)):
            app_args[arg] = app_args[arg].format(
                **test, output_dir=output_dir, socket=args.socket
            )

        print("Cmdline: " + str(app_args))

        try:
            res = subprocess.run(app_args, check=True)
        except Exception as e:
            print("error: " + str(e), file=sys.stderr)

        # terminate telemetry capture process
        if args.telemetry:
            telemetry.stop(args.socket)
            telemetry.kill(args.socket)
            capture_process = processes["telemetry.capture"]["process"]
            if capture_process.is_alive():
                # 30 second timeout should be more than sufficient to join a non-broken process
                capture_process.join(30)
            if capture_process.exitcode == None:
                # If exit code is still zero then we failed to cleanly kill it.
                # Use process API to forcefully kill.
                capture_process.kill()
                print(
                    "error: joining {0} process timed-out".format("telemetry.capture"),
                    file=sys.stderr,
                )
                sys.exit(1)
            if capture_process.exitcode != 0:
                print(
                    "error: {0} process returned non-zero exit code: {1}".format(
                        "telemetry.capture", capture_process.exitcode
                    ),
                    file=sys.stderr,
                )
                sys.exit(1)
            del processes["telemetry.capture"]

        # read in results.json for the current test
        file = os.path.join(output_dir, "results.json")
        with open(file, "r") as f:
            results = json.load(f)

        # incorporate telemetry into results if we collected it
        # Determine name for SMI data and open it
        telemetry_results = {}
        if args.telemetry:
            smi_dump = parse_telemetry.open_smi_data(output_dir, args.platform)
            # Process the SMI data
            telemetry_results = parse_telemetry.process(smi_dump, args.platform)
            # Close SMI CSV
            smi_dump.close()
            # write the telemetry stats to telemetry.json
            file = os.path.join(output_dir, "telemetry.json")
            with open(file, "w") as f:
                json.dump(telemetry_results, f, indent=indent)

        # merge various components into final test report
        report = {}
        report = js_merge.merge(report, results)
        report = js_merge.merge(report, get_config(args.metadata, app_args))
        report = js_merge.merge(report, telemetry_results)
        report = js_merge.merge(report, sysinfo)

        # write the merged report
        file = os.path.join(
            args.output_dir, "results_test_{0}.json".format(iteration + 1)
        )
        with open(file, "w") as f:
            json.dump(report, f, indent=indent)

        # Store non-updated test results for summary
        to_summarize += [results]

    # Write summary CSV
    csv_table = json_to_csv.make_csv(to_summarize)
    file = os.path.join(args.output_dir, "summary.csv")
    json_to_csv.write_table(file, csv_table)
