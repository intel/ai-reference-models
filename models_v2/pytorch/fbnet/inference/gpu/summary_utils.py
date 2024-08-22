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
import copy
import datetime
import torch
import io
import os
import numpy as np
import sys

# sample modules
import io_utils
from arguments_utils import args
try:
    import js_sysinfo
except Exception as e:
    print('fatal: ' + str(e), file=sys.stderr)
    print('fatal: set PYTHONPATH to the location of js_sysinfo.py')
    sys.exit(1)

# Group metric keys per operation to perform when combining results
# calculate min, max, average and stdev of a random value
op_avg = [
    'accuracy-top1',
    'accuracy-top5',
    'latency',
    'total-batches-tested-per-stream',
    'total-images-tested-per-stream'
]
# error out if values are different
op_same = [
    'uniq-batches-tested',
    'uniq-images-tested',
    'warmup-batches'
]
# calculate total min, max, average and stdev of random value
# total means to multiple on a number of streams, i.e. =min(arr)*len(arr)
op_total_avg = [
    'throughput'
]

def get_jit_method():
    if args.jit_trace:
        return 'trace'
    elif args.jit_script:
        return 'script'
    elif args.compile:
        return 'compile'
    return 'no'

# Write single result
def write_results(batches_tested, throughput, latency, top1, top5):
    output_dict = {
        # 'schema' points to json-schema output is compliant to
        # TBD for now, need to replace with URL of the schema
        'schema': 'TBD',
        'config': {
            'workload': {
                'type': 'inference',
                'model': {
                    'name': '{0}'.format(args.arch),
                    'link': 'https://pytorch.org/vision/main/models/generated/torchvision.models.{0}.html'.format(args.arch),
                    'streams': args.total_instances,
                    'precision': args.dtype_str,
                    'batch-size': args.batch_size,
                    'device': args.device,
                    'amp': 'false' if args.no_amp else 'true',
                    'jit': get_jit_method(),
                    'dummy': 'true' if args.dummy else 'false',
                    'framework': 'PyTorch'
                },
            },
            'system': js_sysinfo.get_system_config(all=True, quiet=True),
        },
        'results': {
            'metadata': {
                'date': datetime.datetime.now().isoformat(),
                'tester': ''
            },
            'metrics': {
                'throughput': {
                    'avg': float(throughput),
                    'min': float(throughput),
                    'max': float(throughput),
                    'stdev': 0.0,
                    'units': 'images/s'
                    },
                'accuracy-top1': {
                    'avg': float(top1),
                    'min': float(top1),
                    'max': float(top1),
                    'stdev': 0.0,
                    'units': '%'
                },
                'accuracy-top5': {
                    'avg': float(top5),
                    'min': float(top5),
                    'max': float(top5),
                    'stdev': 0.0,
                    'units': '%'
                },
                'latency': {
                    'avg': float(latency),
                    'min': float(latency),
                    'max': float(latency),
                    'stdev': 0.0,
                    'units': 'ms'
                },
                'total-batches-tested-per-stream': {
                    'avg': float(batches_tested),
                    'min': float(batches_tested),
                    'max': float(batches_tested),
                    'stdev': 0.0,
                    'units': 'ms'
                },
                'total-images-tested-per-stream': {
                    'avg': float(batches_tested * args.batch_size),
                    'min': float(batches_tested * args.batch_size),
                    'max': float(batches_tested * args.batch_size),
                    'stdev': 0.0,
                    'units': 'ms'
                },
                'uniq-batches-tested': { 'total': args.num_inputs // args.batch_size },
                'uniq-images-tested': { 'total': args.num_inputs },
                'warmup-batches': { 'total': args.warm_up },
            }
        }
    }

    io_utils.write_json('{0}/results_{1}.json'.format(args.output_dir, args.instance), output_dict)

def show_test_conditions():
    io_utils.write_info('------------ start inference arguments --------------------------')
    io_utils.stdout_helper('  [MODEL]')
    io_utils.stdout_helper('    running model:      {0}'.format(args.arch))
    io_utils.stdout_helper('    pretrained model:   {0}'.format(args.pretrained))
    io_utils.stdout_helper('    device:             {0}'.format(args.device))
    io_utils.stdout_helper('    model input size:   ({0}, {1})'.format(args.width, args.height))
    io_utils.stdout_helper('    model load:         {0}'.format(args.load))
    io_utils.stdout_helper('    model save:         {0}'.format(args.save))
    io_utils.stdout_helper('  [INPUT-OUTPUT]')
    io_utils.stdout_helper('    output dir:         {0}'.format(args.output_dir))
    if args.dummy:
        io_utils.stdout_helper('    using dummy data:   {0}'.format(True))
    else:
        io_utils.stdout_helper('    val data dir:       {0}'.format(args.data))
    if args.socket:
        io_utils.stdout_helper('    socket:             {0}'.format(args.socket))
    io_utils.stdout_helper('  [DATA TYPE]')
    io_utils.stdout_helper('    using dtype:        {0}'.format(args.dtype_str))
    if args.dtype_str not in ['float32', 'tfloat32', 'bfloat32']:
        io_utils.stdout_helper('    using amp:          {0}'.format(not args.no_amp))
    io_utils.stdout_helper('  [PERF ARGS]')
    io_utils.stdout_helper('    JIT method:         {0}'.format(get_jit_method()))
    io_utils.stdout_helper('    gradients:          {0}'.format('none' if args.no_grad else 'zero' if args.zero_grad else 'true'))
    io_utils.stdout_helper('  [BENCHMARK PARAMS]')
    io_utils.stdout_helper('    warm up batches:    {0}'.format(args.warm_up))
    io_utils.stdout_helper('    batch size:         {0}'.format(args.batch_size))
    io_utils.stdout_helper('    num inputs:         {0}'.format(args.num_inputs))
    io_utils.stdout_helper('    label smoothing:    {0}'.format(args.label_smoothing))
    io_utils.stdout_helper('    channels last:      {0}'.format(args.channels_last))
    io_utils.stdout_helper('    instance info:      {0}/{1}'.format(args.instance, args.total_instances))
    io_utils.stdout_helper('    min test duration:  {0}'.format(args.min_test_duration))
    io_utils.stdout_helper('    max test duration:  {0}'.format(args.max_test_duration))
    io_utils.stdout_helper('  [MISC]')
    io_utils.stdout_helper('    seed:               {0}'.format(args.seed))
    io_utils.stdout_helper('    non-blocking load:  {0}'.format(args.non_blocking))
    io_utils.stdout_helper('    print frequency:    {0}'.format(args.print_frequency))
    io_utils.stdout_helper(' --------------------------- end inference arguments ---------------------------')

def get_valid_results_list():
    results_list = []
    for root, dirs, files in os.walk(args.output_dir):
        for instance in range(1, args.total_instances + 1):
            file = 'results_{0}.json'.format(instance)
            if file in files:
                results_list += [root + os.sep + file]
        break
    io_utils.write_info('Found {0} results to combine'.format(len(results_list)))

    return results_list

def write_yaml_summary(summary):
    yaml_file = io.open(args.output_dir + os.sep + 'results.yaml', 'w')

    yaml_output = []
    yaml_output += ['']
    yaml_output += ['results:']
    yaml_output += [' - key: throughput']
    yaml_output += ['   value: {0}'.format(summary['results']['metrics']['throughput']['avg'])]
    yaml_output += ['   unit: img/s']
    yaml_output += [' - key: latency']
    yaml_output += ['   value: {0}'.format(summary['results']['metrics']['latency']['avg'])]
    yaml_output += ['   unit: ms']
    yaml_output += [' - key: accuracy']
    yaml_output += ['   value: {0}'.format(summary['results']['metrics']['accuracy-top1']['avg'])]
    yaml_output += ['   unit: percents']
    yaml_output += ['']

    yaml_file.write('\n'.join(yaml_output))
    yaml_file.close()
    io_utils.write_info('Wrote {0}'.format(args.output_dir + os.sep + 'results.yaml'))

def combine_results():
    io_utils.write_info('Combining results...')
    result_list = get_valid_results_list()

    status = 'passed'
    summary = {}
    to_avg = {}

    if len(result_list) < args.total_instances:
        io_utils.write_warning('Number of found results ({0}) to combine is less than expected ({1})'.format(len(result_list), args.total_instances))
        # Marking overall summary as belonging to failing case
        status = 'failed'

    for result_path in result_list:
        result = io_utils.read_json(result_path)
        if not summary:
            summary = copy.deepcopy(result)
            continue
        if summary['config'] != result['config']:
            io_utils.write_warning('Incompatible config: {0}'.format(result_path))
            # Marking overall summary as belonging to failing case
            status = 'failed'
            continue
        if summary['results'].keys() != result['results'].keys():
            io_utils.write_warning('Different set of keys in config: {0}'.format(result_path))
            # Marking overall summary as belonging to failing case
            status = 'failed'
        for key in result['results']['metrics']:
            if key in op_same:
                if summary['results']['metrics'][key] != result['results']['metrics'][key]:
                    io_utils.write_warning('Mismatched metric value(s) while expecting the same: key={0}, config={1}'.format(key, result_path))
                    # Marking overall summary as belonging to failing case
                    status = 'failed'
            elif key in op_avg or key in op_total_avg:
                if key not in to_avg:
                    to_avg[key] = [summary['results']['metrics'][key]['avg']]
                to_avg[key].append(result['results']['metrics'][key]['avg'])
            else:
                # We should not be here, that's a bug: we forgot to classify a key
                io_utils.write_warning('BUG: unclassified key: key={0}, config={1}'.format(key, result_path))
                status = 'failed'

    # calculating min/max/sums/stdev
    for key in to_avg:
        nstreams = len(to_avg[key])
        summary['results']['metrics'][key]['min'] = min(to_avg[key])
        summary['results']['metrics'][key]['max'] = max(to_avg[key])
        summary['results']['metrics'][key]['avg'] = np.mean(np.array(to_avg[key], dtype=np.float64))
        summary['results']['metrics'][key]['stdev'] = np.std(np.array(to_avg[key], dtype=np.float64))
        if key in op_total_avg:
            summary['results']['metrics'][key]['min'] *= nstreams
            summary['results']['metrics'][key]['max'] *= nstreams
            summary['results']['metrics'][key]['avg'] *= nstreams
            summary['results']['metrics'][key]['stdev'] *= nstreams

    # setting overall status
    summary['results']['metrics']['status'] = status

    io_utils.write_json(args.output_dir + os.sep + 'results.json', summary)
    io_utils.write_info('Wrote {0}'.format(args.output_dir + os.sep + 'results.json'))

    write_yaml_summary(summary)
    io_utils.write_info('Wrote {0}'.format(args.output_dir + os.sep + 'results.yaml'))
