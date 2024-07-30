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
import time
import json
import os
import io
import torch
import datetime
import numpy as np
from numpy import inf
import copy
import sys

# sample modules
import io_utils

try:
    import js_sysinfo
    from js_merge import merge
except Exception as e:
    io_utils.write_error(str(e))
    io_utils.write_error('set PYTHONPATH to the location of js_sysinfo.py')
    sys.exit(1)

# Group metric keys per operation to perform when combining results
# calculate min, max, average and stdev of a random value
op_avg = [
    'latency',
    'accuracy',
]

#Keys expected to be same across reports
op_same = [
    'warmup-batches'
]

# calculate total min, max, average and stdev of random value
# total means to multiple on a number of streams, i.e. =min(arr)*len(arr)
op_total_avg = [
    'throughput',
    'total-batches-tested',
    'total-images-tested',
    'uniq-batches-tested',
    'uniq-images-tested',
]

MILLISECONDS = 1000

def write_instance_result(args, reports, tag=None):
    #Create Json dump of results for one instance
    merged_report = {}
    for d in reports:
        merged_report = merge(d, merged_report)

    # Dump all fields in a json report
    file_name="results{suffix}.json".format(suffix = "" if tag is None else "_" + str(tag))
    json_path= os.path.join(args.output, file_name)
    with io.open(json_path, "w", encoding="utf-8") as file_object:
        json.dump(merged_report, file_object, indent=4)

def write_yaml_result(args, summary):
    # Write key fields in a yaml report across all instances
    yaml_path = os.path.join(args.output, 'results.yaml')
    with io.open(yaml_path, "w", encoding="utf-8") as yaml_file:
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
        yaml_output += ['   value: {0}'.format(summary['results']['metrics']['accuracy']['avg'])]
        yaml_output += ['   unit: percents']
        yaml_output += ['']
        yaml_file.write('\n'.join(yaml_output))

def get_valid_results_list(args):
    results_list = []
    for root, dirs, files in os.walk(args.output):
        for instance in range(1, args.streams + 1):
            file="results{suffix}.json".format(suffix = "_" + str(instance))
            if file in files:
                results_list += [root + os.sep + file]
        break
    io_utils.write_info('Found {0} results to combine'.format(len(results_list)))
    return results_list

def combine_results(args):
    summary = {}
    status = "passed"
    to_avg = {}
    streams = args.streams
    results_list = get_valid_results_list(args)

    for result_path in results_list:
        try:
            with io.open(result_path, 'r', encoding='utf-8') as file_object:
                result = json.load(file_object)
        except:
            io_utils.write_warning(f'Failed to read {result_path} as a json object. Skipping processing')
            continue

        if not summary:
            summary = copy.deepcopy(result)
            continue
        if summary['config'] != result['config']:
            io_utils.write_warning('Incompatible config: {0}'.format(result_path))
            # Marking overall summary as belonging to failing case
            status = "failed"
            continue
        if summary['results'].keys() != result['results'].keys():
            io_utils.write_warning('Different set of keys in config: {0}'.format(result_path))
            # Marking overall summary as belonging to failing case
            status = 'failed'

        for key in result['results']['metrics']:
            if key in op_same:
                if summary['results']['metrics'][key] != result['results']['metrics'][key]:
                    io_utils.write_warning('Mismatched metric value(s) while expecting the same: key={0}, config={1}'.format(key, result_path))
            elif key in op_avg or key in op_total_avg:
                if key not in to_avg:
                    to_avg[key] = [summary['results']['metrics'][key]['avg']]
                to_avg[key].append(result['results']['metrics'][key]['avg'])
            else:
                # We should not be here, that's a bug: we forgot to classify a key
                io_utils.write_warning('Unclassified key: key={0}, config={1}'.format(key, result_path))
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
    if 'results' in summary.keys() and 'metrics' in summary['results'].keys():
        summary['results']['metrics']['status'] = status

    # Write summary json using write_instance_result with None tag
    json_path= os.path.join(args.output, "results.json")
    with io.open(json_path, "w", encoding="utf-8") as file_object:
        json.dump(summary, file_object, indent=4)

    # Write yaml summary of key results
    try:
        write_yaml_result(args, summary)
    except Exception as e:
        io_utils.write_error('Failure when writing yaml summary of results - ' + type(e).__name__ + ":" +  str(e))


def get_env_report(args, model):
    # Set defaults reflecting current implementation status for following flags
    # In future, these may be updated when the feature is supported in the sample
    batch_size = 1
    batch_streaming = 1
    amp = "true" if args.amp else "false"
    jit = "none"

    def get_precision_string(type):
        mapping = { "bf16": "bfloat16", "fp16": "float16", "fp32": "float32" }
        def_val = "unknown:" + str(type)
        precision_string = mapping.get(type, def_val)
        return precision_string

    def get_test_name(args):
        test_name =  args.device + "-" + str(args.streams) + "pro-" + args.precision
        if args.amp:
            test_name += "-amp"
        if not args.dummy:
            test_name += "-dataset"
        return test_name

    output_dict = {
        "schema" : "TBD",
        "config" : {
            "metadata" : {
                "name" : "rife-" + get_test_name(args),
                "version" : "TBD",
            },
            "workload" : {
                "type" : "inference",
                "name" : get_test_name(args),
                "source" : {
                    "github" : "TBD",
                },
                "model" : {
                    "name": "IFRNet",
                    "link": "https://github.com/ltkong218/IFRNet/tree/b117bcafcf074b2de756b882f8a6ca02c3169bfe",
                    "streams": args.streams,
                    "precision" : get_precision_string(args.precision),
                    "batch-size": batch_size,
                    "device" : args.device,
                    "amp": amp,
                    "jit": jit,
                    "dummy": "true" if args.dummy else "false",
                    "framework": "PyTorch",
                },
                "dataset" : {
                    "channels": args.ddim[0],
                    "height": args.ddim[1],
                    "width": args.ddim[2]
                }
            },
            "system" : js_sysinfo.get_system_config(all=True, quiet=True),
        },
    }
    return output_dict

class PerfManager:
    def __init__(self, args):
        self.args = args
        self.clear_timer()
        self._init_time = time.perf_counter()
        self._reports = []

    #Functions covering a single iteration
    def start_timer(self):
        self._start = time.perf_counter()

    def end_timer(self):
        self._end = time.perf_counter()

    def time_elapsed(self):
        return time.perf_counter() - self._start

    def check_timer_elapsed(self, target_in_s=inf):
        return True if self.time_elapsed() >= target_in_s else False

    def gather_metrics(self, frame_count, batch_size=1, mode="new", display=None, quality_metrics=None):
        batch_count = frame_count / batch_size
        duration_s = self._end - self._start
        duration_ms = duration_s * MILLISECONDS
        avg_latency = duration_ms / batch_count
        avg_fps = frame_count / duration_s
        try:
            accuracy = quality_metrics["results"]["metrics"]["accuracy"]["avg"]
        except:
            accuracy = None

        report_dict = {
            #Timestamps in seconds, latencies in milliseconds
            "init_time": self._init_time,
            "start_time" : self._start,
            "end_time" : self._end,
            "report_time": time.perf_counter(),
            "frame_count" : frame_count,
            "index" : len(self._reports),
            "batch_size" : batch_size,
            "duration" : duration_ms,
            "avg_latency" : avg_latency,
            "avg_fps" : avg_fps,
            "misc"  : None,
            "accuracy" : accuracy
        }
        self.report_metrics(report_dict, frame_count=frame_count, mode=mode, display=display)
        return report_dict

    def report_metrics(self, report_dict, frame_count, mode="all", display=None):
        def print_report(report):
            if not display:
                return
            display_string = "{ " + display +" }"
            frames = report["frame_count"]
            avg_tp = "{:3.3f}".format(report["avg_fps"])
            avg_latency = "{:3.3f}".format(report["avg_latency"])
            if self.args.dummy:
                accuracy = "NA"
            elif report["accuracy"] == None:
                accuracy = "NA"
            else:
                accuracy = "{:3.3f}%".format(report["accuracy"])

            # Calculate how long we have ran the benchmark so far
            current_benchmark_duration = self.time_elapsed()
            # Estimate how long a single pass through the dataset takes
            dataset_time_estimate = current_benchmark_duration * (frame_count / frames)
            estimated_total_duration = dataset_time_estimate
            if self.args.dummy:
                # Estimated total duration must be a multiple of complete passes through the dataset that exceeds min test duration
                while estimated_total_duration < self.args.min_test_duration:
                    estimated_total_duration += dataset_time_estimate
                # Estimated total duration is hard capped at max test duration
                if estimated_total_duration > self.args.max_test_duration:
                    estimated_total_duration = self.args.max_test_duration
            else:
                # Total estimate is current latency (in seconds) times number of total batches (latency is per batch).
                estimated_total_duration = (report["avg_latency"] / 1000) * frame_count

            timestamp = '~{0:5.1f}% after {1:.2f}s (Estimated {2:3.0f}s remaining)'.format(
                min([100, 100 if estimated_total_duration == 0 else 100 * current_benchmark_duration / estimated_total_duration]),
                current_benchmark_duration,
                max([0, estimated_total_duration - current_benchmark_duration])
            )

            io_utils.write_info(f"[{timestamp}] PERF_STATUS: frames = {frames}, avg_throughput = {avg_tp} frames/s, avg_latency = {avg_latency} ms, avg_accuracy = {accuracy}. " + display_string)

        if mode == "skip":
            print_report(report_dict)
        elif mode == "update":
            if not self._reports:
                self._reports.append(report_dict)
                self._reports[0]["index"] = 0
            else:
                self._reports[0]["end_time"] = report_dict["end_time"]
                self._reports[0]["report_time"] = report_dict["report_time"]
                self._reports[0]["frame_count"] = report_dict["frame_count"]
                self._reports[0]["duration"] += report_dict["duration"]
                self._reports[0]["accuracy"] = report_dict["accuracy"]
                batch_count = self._reports[0]["frame_count"] / self._reports[0]["batch_size"]
                avg_fps = MILLISECONDS * self._reports[0]["frame_count"] / self._reports[0]["duration"]
                avg_latency =  self._reports[0]["duration"] / batch_count
                self._reports[0]["avg_fps"] = avg_fps
                self._reports[0]["avg_latency"] = avg_latency
                self._reports[0]["index"] += 1
            print_report(self._reports[0])
        elif mode == "new":
            self._reports = [report_dict]
            self._reports[0]["index"] = 0
            print_report(report_dict)
        else:
            report_dict["index"] = len(self._reports)
            self._reports.append(report_dict)
            print_report(self._reports[-1])

    def clear_timer(self):
        self._start = None
        self._end = None

    def restart_timer(self):
        self.clear_timer()
        self.start_timer()

    #Summarize across iterations (usually = 1)
    def summarize_perf(self, warmup_report=None):
        frame_counts = [ report["frame_count"] for report in self._reports ]
        latency      = [ report["avg_latency"] for report in self._reports ]
        fps          = [ report["avg_fps"] for report in self._reports ]

        summary_avg_fps = sum(fps)/len(fps)
        summary_avg_latency = sum(latency)/len(latency)

        batch_streaming = 1
        batch_size = self._reports[0]["batch_size"]
        wallclock_time = self._reports[-1]["end_time"] - self._reports[0]["start_time"]
        wallclock_fps = sum(frame_counts) / wallclock_time
        wallclock_latency = wallclock_time * batch_size * MILLISECONDS / sum(frame_counts)

        warmup_batches = None
        if warmup_report is not None:
            try:
                warmup_batches = warmup_report["warmup_result"]["results"]["metrics"]["total-batches-tested"]
            except KeyError:
                io_utils.write_warning("Unexpected error in reading warmup report. Reporting as 0")

        if warmup_batches is None:
            warmup_batches = { "avg" : 0, "units": "batches" }

        report_dict = {
            "results": {
                'metadata': {
                    'date': datetime.datetime.now().isoformat(),
                    'tester': ''
                },
                "metrics" : {
                    "latency" : {
                        "avg" : summary_avg_latency,
                        "min" : summary_avg_latency,
                        "max" : summary_avg_latency,
                        "stdev" : 0,
                        "units" : "ms"
                    },
                    "throughput" : {
                        "avg" : summary_avg_fps,
                        "min" : summary_avg_fps,
                        "max" : summary_avg_fps,
                        "stdev" : 0,
                        "units" : "images/s"
                    },
                    "total-batches-tested": {
                        "avg" : sum(frame_counts) / batch_size,
                        "min" : sum(frame_counts) / batch_size,
                        "max" : sum(frame_counts) / batch_size,
                        "stdev" : 0,
                        "units" : "batches"
                    },
                    "uniq-batches-tested": {
                        "avg" : sum(frame_counts) / (batch_size * batch_streaming),
                        "min" : sum(frame_counts) / (batch_size * batch_streaming),
                        "max" : sum(frame_counts) / (batch_size * batch_streaming),
                        "stdev" : 0,
                        "units" : "batches"
                    },
                    "total-images-tested": {
                        "avg" : sum(frame_counts),
                        "min" : sum(frame_counts),
                        "max" : sum(frame_counts),
                        "stdev" : 0,
                        "units" : "images"
                    },
                    "uniq-images-tested": {
                        "avg" : sum(frame_counts) / batch_streaming,
                        "min" : sum(frame_counts) / batch_streaming,
                        "max" : sum(frame_counts) / batch_streaming,
                        "stdev" : 0,
                        "units" : "images"
                    },
                    "warmup-batches": warmup_batches,
                }
            },
        }
        return report_dict

    def reset(self):
        self.clear_timer()
        self._reports = []
