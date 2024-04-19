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
import os

import mlperf_loadgen as lg

import logging
from SUT import SUT

from utils import getArgs

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("GPT-J")

SCENARIO_MAP = {
    "singlestream": lg.TestScenario.SingleStream,
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}



def main():
    args = getArgs()

    settings = lg.TestSettings()
    scenario = args.scenario

    settings.scenario = SCENARIO_MAP[args.scenario.lower()]
    settings.FromConfig(args.mlperf_conf, args.workload_name, args.scenario)
    settings.FromConfig(args.user_conf, args.workload_name, args.scenario)

    settings.mode = lg.TestMode.AccuracyOnly if args.mode.lower()=="accuracy" else lg.TestMode.PerformanceOnly

    if args.workers_proc_alloc:
        args.workers_proc_alloc = [int(workers) for workers in args.workers_proc_alloc.split(",")]

    if args.batch_proc_alloc:
        args.batch_proc_alloc = args.batch_proc_alloc.split(";")

    sut = SUT(args.num_proc, args.cpus_per_proc, args.model_checkpoint_path, initial_core=args.cores_offset, batch_size=args.batch_size, dataset_path=args.dataset_path, 
            workers_per_proc=args.workers_per_proc, warmup=args.warmup, precision=args.precision, quantized_model=args.quantized_model, total_sample_count=args.total_sample_count, pad_inputs=args.pad_inputs,
            bind_logical_cores=args.bind_logical_cores, logical_cores_start=args.logical_cores_start,
            workers_proc_alloc=args.workers_proc_alloc, batch_proc_alloc=args.batch_proc_alloc)

    # Start SUT
    sut.startSUT()

    # Create SUT, QSL Trampoline
    lg_sut = lg.ConstructSUT(sut.issueQueries, sut.flushQueries)
    lg_qsl = lg.ConstructQSL(args.total_sample_count, args.total_sample_count, sut.loadSamplesToRam, sut.unloadSamplesFromRam)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = False

    # Start loadgen test
    log.info("Starting {}-{} Test".format(args.scenario, args.mode))
    lg.StartTestWithLogSettings(lg_sut, lg_qsl, settings, log_settings)

    log.info("Test completed")
    # Stop SUT
    sut.stopSUT()

    lg.DestroyQSL(lg_qsl)
    lg.DestroySUT(lg_sut)

if __name__=="__main__":
    main()
