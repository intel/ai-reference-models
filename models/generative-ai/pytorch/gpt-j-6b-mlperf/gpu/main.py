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
# ============================================================================

import arguments
import logging
import mlperf_loadgen as lg
import os

from pytorch_sut import OfflineSUT, ServerSUT

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("GPT-J")

scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def main():
    args = arguments.parse_args()
    if args.scenario == "Server":
        sut = ServerSUT(
            args.model_path, args.dataset_path, args.dtype, args.device,
            args.num_workers, args.num_beams, args.scenario, args
        )
    else:
        sut = OfflineSUT(
            args.model_path, args.dataset_path, args.dtype, args.device,
            args.num_workers, args.num_beams, args.scenario, args
        )
    lg_qsl = lg.ConstructQSL(
        sut.dataset.count,
        sut.dataset.perf_count,
        sut.dataset.LoadSamplesToRam,
        sut.dataset.UnloadSamplesFromRam
    )
    lg_sut = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    # set cfg
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "gptj", args.scenario)
    settings.FromConfig(args.user_conf, "gptj", args.scenario)
    settings.mode = (
        lg.TestMode.AccuracyOnly if args.accuracy else lg.TestMode.PerformanceOnly
    )
    # set log
    os.makedirs(args.log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = True
    # run benchmark
    print("==> Running loadgen test")
    lg.StartTestWithLogSettings(lg_sut, lg_qsl, settings, log_settings, args.audit_conf)
    print("Done!")


if __name__ == "__main__":
    main()
