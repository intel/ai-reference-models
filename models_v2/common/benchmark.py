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
import json
import os
import pathlib
import shutil
import subprocess
import sys

# Custom modules
import js_merge
import js_sysinfo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='benchmark',
          description='Benchmark application given with templated cmdline arguments',
          epilog='Copyright (c) 2024 Intel Corporation')

    #parser.add_argument('-v', '--verbose', action="store_true", help='Enable verbose output')
    #parser.add_argument('-q', '--quiet', action="store_true", help='Be quiet and suppress messages to stdout')

    parser.add_argument('--indent', default=None, help='indent for json.dump()')
    parser.add_argument('--output_dir', action="store", type=str, default='', required=True, help='Path to store outputs')
    parser.add_argument('--profile', action="store", type=str, default='', required=True, help='Profile with tests list')

    args, benchmark_app_args = parser.parse_known_args()

    indent = int(args.indent) if args.indent and args.indent.isdecimal() else args.indent

    # getting tests list from given profile
    tests=[]
    with open(args.profile, 'r') as f:
        ps=csv.DictReader(f)
        for p in ps:
            tests += [p]

    print('Profile: ' + args.profile)
    print('Tests: {}'.format(len(tests)))

    # configuring output directory
    # ultimately we will have the following layout of output directory:
    #
    # .
    # ├── profile.csv               # copy of profile
    # ├── sysinfo.json              # baremetal sysinfo description
    # ├── test_{i}                  # output directory of i-th test
    # │   ├── test.csv              # test csv definition (i-th line from profile.csv)
    # │   ├── results.json          # results output from the test complying with the schema
    # │   └── *                     # whatever other output files test produces
    # └── results_test_{i}.json     # ultimate report for i-th test
    #
    # Fail if can't create the directory or if it exists.
    pathlib.Path(args.output_dir).mkdir()

    # copy the whole profile
    shutil.copyfile(args.profile, os.path.join(args.output_dir, 'profile.csv'))

    # get system description
    sysinfo_parser = js_sysinfo.get_parser()
    sysinfo_args = sysinfo_parser.parse_args(['--all', '--verbose'])
    sysinfo = js_sysinfo.get_sysinfo(**vars(sysinfo_args))

    # dump system description to .json file
    file = os.path.join(args.output_dir, 'sysinfo.json')
    with open(file, 'w') as f:
        json.dump(sysinfo, f, indent=indent)

    for i, t in enumerate(tests):
        print('Running: {0}/{1}'.format(i+1, len(tests)))

        # create output directory for the test
        output_dir = os.path.join(args.output_dir, 'test_' + str(i+1))
        pathlib.Path(output_dir).mkdir()

        # dump test csv definition
        with open(os.path.join(output_dir, 'test.csv'), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=t.keys())
            writer.writeheader()
            writer.writerow(t)

        # creating benchmark app cmdline from the template
        app_args = list(benchmark_app_args)
        for a in range(len(app_args)):
            app_args[a] = app_args[a].format(**t, output_dir=output_dir, control_file='tbd')

        print('Cmdline: ' + str(app_args))

        try:
            res = subprocess.run(app_args, check=True)
        except Exception as e:
            print('error: ' + str(e), file=sys.stderr)

        # read in results.json for the current test
        file = os.path.join(output_dir, 'results.json')
        try:
            with open(file, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print('error: ' + str(e), file=sys.stderr)
            sys.exit(1)

        # merge sysinfo with test results
        report = js_merge.merge(results, sysinfo)

        # write the merged report
        file = os.path.join(args.output_dir, 'results_test_{0}.json'.format(i+1))
        with open(file, 'w') as f:
            json.dump(report, f, indent=indent)
