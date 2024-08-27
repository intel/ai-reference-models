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
import torch
import io
import os
import numpy as np

# sample modules
import io_utils
from arguments_utils import args

def get_jit_method():
    if args.use_jit == 'trace':
        return 'trace'
    elif args.use_jit == 'script':
        return 'script'
    elif args.use_jit == 'compile':
        return 'compile'
    return 'no'

def write_results(batches_tested, throughput, latency, overall, whole, core, enhancing, throughput_overhead, latency_overhead):
    output_dict = {
        'SAME:Model': '{0}'.format(args.arch),
        'SAME:Framework': 'PyTorch',
        'SAME:Streams': args.total_instances,
        'SAME:Batchsize': args.batch_size,
        'SAME:Total Batches Tested': batches_tested,
        'SAME:Unique Batches Tested': batches_tested // args.batch_streaming,
        'SAME:Total Images Tested': batches_tested * args.batch_size,
        'SAME:Unique Images Tested': args.batch_size * (batches_tested // args.batch_streaming),
        'SAME:Warmup Batches': args.warm_up,
        'SAME:Data Type': args.dtype_str,
        'SAME:AMP': 'no' if args.no_amp else 'yes',
        'SAME:JIT': get_jit_method(),
        'MAX:Latency (ms)': float(latency),
        'MAX:Latency with Overhead (ms)': float(latency_overhead),
        'SUM_MIN:Throughput (img/s)': float(throughput),
        'SUM_MIN:Throughput with Overhead (img/s)': float(throughput_overhead),
        'MIN:Total Accuracy (%)': float(overall),
        'MIN:Whole Tumor Accuracy (%)': float(whole),
        'MIN:Core Tumor Accuracy (%)': float(core),
        'MIN:Enhancing Tumor Accuracy (%)': float(enhancing),
        'SAME:Device': args.device,
        'AND:Pass': True
    }
    output_dict['SAME:Framework Metadata'] = {}
    output_dict['SAME:Framework Metadata']['torch_version'] = str(torch.__version__)
    if args.xpu and args.ipex:
        import intel_extension_for_pytorch as ipex
        output_dict['SAME:Framework Metadata']['ipex_version'] = str(ipex.__version__)
        output_dict['SAME:Framework Metadata']['has_onemkl'] = ipex.xpu.has_onemkl()
    io_utils.write_json('{0}/results_{1}.json'.format(args.output_dir, args.instance), output_dict)

def show_test_conditions():
    io_utils.write_info('------------ start inference arguments --------------------------')
    io_utils.stdout_helper('  [MODEL]')
    io_utils.stdout_helper('    running model:      {0}'.format(args.arch))
    io_utils.stdout_helper('    device:             {0}'.format(args.device))
    io_utils.stdout_helper('    model input size:   (4, 224, 224, 160)')
    io_utils.stdout_helper('  [INPUT-OUTPUT]')
    io_utils.stdout_helper('    output dir:         {0}'.format(args.output_dir))
    if args.dummy:
        io_utils.stdout_helper('    using dummy data :  {0}'.format(True))
    else:
        io_utils.stdout_helper('    val data dir:       {0}'.format(args.data))
        io_utils.stdout_helper('    GT Label dir:       {0}'.format(args.label_data_dir))
    io_utils.stdout_helper('  [DATA TYPE]')
    io_utils.stdout_helper('    using dtype:        {0}'.format(args.dtype_str))
    if args.dtype_str not in ['float32', 'tfloat32', 'bfloat32']:
        io_utils.stdout_helper('    using amp:          {0}'.format(not args.no_amp))
    if args.dtype_str in ['int8', 'uint8']:
        quantization_method = '{0} {1} quantization'.format(
            'perchannel weights' if args.perchannel_weight else 'non-perchannel weights',
            'asymmetric' if args.asymmetric_quantization else 'symmetric'
        )
        io_utils.stdout_helper('    quantization method: {0}'.format(quantization_method))
        io_utils.stdout_helper('    calib iters:         {0}'.format(args.calib_iters))
        io_utils.stdout_helper('    calib batch size:    {0}'.format(args.calib_bs))
    io_utils.stdout_helper('  [PERF ARGS]')
    io_utils.stdout_helper('    JIT method:         {0}'.format(get_jit_method()))
    io_utils.stdout_helper('    gradients:          zero')
    io_utils.stdout_helper('  [BENCHMARK PARAMS]')
    io_utils.stdout_helper('    warm up batches:    {0}'.format(args.warm_up))
    io_utils.stdout_helper('    batch size:         {0}'.format(args.batch_size))
    io_utils.stdout_helper('    repeat batches:     {0}'.format(args.batch_streaming))
    io_utils.stdout_helper('    max data set size:  {0}'.format(args.max_val_dataset_size))
    io_utils.stdout_helper('    channels last:      {0}'.format(args.channels_last))
    io_utils.stdout_helper('    instance info:      {0}/{1}'.format(args.instance, args.total_instances))
    io_utils.stdout_helper('  [MISC]')
    io_utils.stdout_helper('    non-blocking load:  {0}'.format(args.non_blocking))
    io_utils.stdout_helper('    status prints:      {0}'.format(args.status_prints))
    io_utils.stdout_helper(' --------------------------- end inference arguments ---------------------------')

def combine_results():
    pre_processed_summary = {}
    io_utils.write_info('Combining results...')
    results = []
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if 'results_' in file and '.json' in file:
                results += [root + os.sep + file]
    io_utils.write_info('Found {0} results to combine'.format(len(results)))
    if len(results) < args.total_instances:
        io_utils.write_info('Number of found results ({0}) to combine is less than expected ({1})'.format(len(results), args.total_instances))
        if 'AND' not in pre_processed_summary:
            pre_processed_summary['AND'] = {}
        pre_processed_summary['AND']['Pass'] = False

    for result_path in results:
        result = io_utils.read_json(result_path)
        for key in result:
            split_key = key.split(':')
            if len(split_key) != 2:
                io_utils.write_warning('Key "{0}" in "{1}" does not match required format of [TYPE]:[NAME]'.format(key, result_path))
                io_utils.write_warning('Key "{0}" will be ignored'.format(key))
                continue
            allowed_key_types = ['AVG', 'SAME', 'SUM', 'SUM_MIN', 'SUM_MAX', 'MIN', 'MAX', 'AND', 'OR']
            if split_key[0] not in allowed_key_types:
                io_utils.write_warning('Key "{0}" in "{1}" does has a valid type'.format(key, result_path))
                io_utils.write_warning('Valid types are: {0}'.format(', '.join(allowed_key_types)))
                io_utils.write_warning('Key "{0}" will be ignored'.format(key))
                continue
            if len(split_key[1]) == 0:
                io_utils.write_warning('Key "{0}" in "{1}" cannot have a name of length zero'.format(key, result_path))
                io_utils.write_warning('Key "{0}" will be ignored'.format(key))
                continue
            key_type = split_key[0]
            key_name = split_key[1]
            value = result[key]

            if key_type not in pre_processed_summary:
                pre_processed_summary[key_type] = {}

            if key_type in ['AVG', 'SUM', 'SUM_MIN', 'SUM_MAX', 'MIN', 'MAX']:
                if key_name not in pre_processed_summary[key_type]:
                    pre_processed_summary[key_type][key_name] = []
                pre_processed_summary[key_type][key_name] += [float(value)]
            elif key_type == 'SAME':
                pre_processed_summary[key_type][key_name] = value
            elif key_type == 'AND':
                if key_name not in pre_processed_summary[key_type]:
                    pre_processed_summary[key_type][key_name] = True
                pre_processed_summary[key_type][key_name] &= value
            elif key_type == 'OR':
                if key_name not in pre_processed_summary[key_type]:
                    pre_processed_summary[key_type][key_name] = False
                pre_processed_summary[key_type][key_name] |= value

    summary = {}
    for key_type in pre_processed_summary:
        for key in pre_processed_summary[key_type]:
            if key_type == 'AVG':
                summary[key] = sum(pre_processed_summary[key_type][key]) / len(pre_processed_summary[key_type][key])
                summary[key + ' - STDEV'] = np.std(np.array(pre_processed_summary[key_type][key], dtype=np.float64))
            elif key_type == 'SAME':
                summary[key] = pre_processed_summary[key_type][key]
            elif key_type == 'SUM':
                summary[key] = sum(pre_processed_summary[key_type][key])
                summary[key + ' - STDEV'] = np.std(np.array(pre_processed_summary[key_type][key], dtype=np.float64))
            elif key_type == 'SUM_MIN':
                summary[key] = min(pre_processed_summary[key_type][key]) * len(pre_processed_summary[key_type][key])
                summary[key + ' - STDEV'] = np.std(np.array(pre_processed_summary[key_type][key], dtype=np.float64))
            elif key_type == 'SUM_MAX':
                summary[key] = max(pre_processed_summary[key_type][key]) * len(pre_processed_summary[key_type][key])
                summary[key + ' - STDEV'] = np.std(np.array(pre_processed_summary[key_type][key], dtype=np.float64))
            elif key_type == 'MIN':
                summary[key] = min(pre_processed_summary[key_type][key])
                summary[key + ' - STDEV'] = np.std(np.array(pre_processed_summary[key_type][key], dtype=np.float64))
            elif key_type == 'MAX':
                summary[key] = max(pre_processed_summary[key_type][key])
                summary[key + ' - STDEV'] = np.std(np.array(pre_processed_summary[key_type][key], dtype=np.float64))
            elif key_type in ['AND', 'OR']:
                summary[key] = pre_processed_summary[key_type][key]

    io_utils.write_json(args.output_dir + os.sep + 'results.json', summary)
    io_utils.write_info('Wrote {0}'.format(args.output_dir + os.sep + 'results.json'))

    yaml_file = io.open(args.output_dir + os.sep + 'results.yaml', 'w')
    yaml_output = []
    yaml_output += ['']
    yaml_output += ['results:']
    yaml_output += [' - key: throughput']
    yaml_output += ['   value: {0}'.format(summary['Throughput (img/s)'])]
    yaml_output += ['   unit: img/s']
    yaml_output += [' - key: latency']
    yaml_output += ['   value: {0}'.format(summary['Latency (ms)'])]
    yaml_output += ['   unit: ms']
    yaml_output += [' - key: accuracy']
    yaml_output += ['   value: {0}'.format(summary['Total Accuracy (%)'])]
    yaml_output += ['   unit: percents']
    yaml_output += ['']
    yaml_file.write('\n'.join(yaml_output))
    yaml_file.close()
    io_utils.write_info('Wrote {0}'.format(args.output_dir + os.sep + 'results.yaml'))
