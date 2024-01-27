# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020 INTEL CORPORATION. All rights reserved.
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
try:
    # FIXME: this is a WA: It will fail in Ubuntu 22.04 if mlperf_loadgen is
    # imported before IPEX.
    import torch
    import intel_extension_for_pytorch
except ImportError:
    pass
import mlperf_loadgen as lg
import subprocess


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend",
                        choices=["pytorch", "onnxruntime", "tf", "ov"],
                        default="pytorch",
                        help="Backend")
    parser.add_argument(
        "--scenario",
        choices=["SingleStream", "Offline", "Server", "MultiStream"],
        default="Offline",
        help="Scenario")
    parser.add_argument("--accuracy",
                        action="store_true",
                        help="enable accuracy pass")
    parser.add_argument("--mlperf_conf",
                        default="build/mlperf.conf",
                        help="mlperf rules config")
    parser.add_argument("--user_conf",
                        default="user.conf",
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument(
        "--model_dir",
        default=
        "build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1",
        help="Path to the directory containing plans.pkl")
    parser.add_argument("--model", help="Path to the ONNX, OpenVINO, or TF model")
    parser.add_argument("--preprocessed_data_dir",
                        default="build/preprocessed_data",
                        help="path to preprocessed data")
    parser.add_argument("--performance_count",
                        type=int,
                        default=16,
                        help="performance count")
    parser.add_argument("--run_fp16",
                        required=False,
                        help="Flag for inference in FP16",
                        action="store_true")
    parser.add_argument("--run_int8",
                        required=False,
                        help="Flag for inference in INT8",
                        action="store_true")
    parser.add_argument("--calib_iters",
                        required=False,
                        type=int,
                        help="number of iterations during calibration",
                        default=1)
    parser.add_argument("--channels_last",
                        required=False,
                        help="Flag to enable chanels-last",
                        action="store_true")
    parser.add_argument("--profiling",
                        required=False,
                        help="Flag to enabling profiling",
                        action="store_true")
    parser.add_argument("--build_dir",
                        default="build",
                        help="Path to the build directory")
    parser.add_argument("--asymm", required=False, action="store_true", help="Flag for asymm quant")
    parser.add_argument("--uint8", required=False, action="store_true", help="Use uint8 dtype for int8 inference")

    args = parser.parse_args()
    return args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream
}


def main():
    args = get_args()

    if args.backend == "pytorch":
        from pytorch_SUT import get_pytorch_sut
        calib_args = None
        if os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]:
                args.profiling = True
        print(args)
        sut = get_pytorch_sut(args.model_dir, args.preprocessed_data_dir,
                              args.performance_count, args.run_fp16,
                              args.run_int8, args.calib_iters,
                              args.channels_last, args.asymm, args.uint8, args.profiling)
    elif args.backend == "onnxruntime":
        from onnxruntime_SUT import get_onnxruntime_sut
        sut = get_onnxruntime_sut(args.model, args.preprocessed_data_dir,
                                  args.performance_count)
    elif args.backend == "tf":
        from tf_SUT import get_tf_sut
        sut = get_tf_sut(args.model, args.preprocessed_data_dir,
                         args.performance_count)
    elif args.backend == "ov":
        from ov_SUT import get_ov_sut
        sut = get_ov_sut(args.model, args.preprocessed_data_dir,
                         args.performance_count)
    else:
        raise ValueError("Unknown backend: {:}".format(args.backend))

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "3d-unet", args.scenario)
    settings.FromConfig(args.user_conf, "3d-unet", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    log_path = os.path.join(args.build_dir, "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    print("Running Loadgen test...")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)

    if args.accuracy and (not args.profiling):
        print("Running accuracy script...")
        accuracy_log_file = os.path.join(args.build_dir, "logs/mlperf_log_accuracy.json")
        accuracy_preprocessed_data = os.path.join(args.build_dir, "preprocessed_data")
        accuracy_postprocessed_data = os.path.join(args.build_dir, "postprocessed_data")
        accuracy_label_data = os.path.join(args.build_dir, "raw_data/nnUNet_raw_data/Task043_BraTS2019/labelsTr")
        cmd = "python3 accuracy-brats.py --log_file {} " \
              "--preprocessed_data_dir {} " \
              "--postprocessed_data_dir {} " \
              "--label_data_dir {}".format(accuracy_log_file,
                                           accuracy_preprocessed_data,
                                           accuracy_postprocessed_data,
                                           accuracy_label_data)
        subprocess.check_call(cmd, shell=True)

    print("Done!")

    print("Destroying SUT...")
    lg.DestroySUT(sut.sut)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl.qsl)


if __name__ == "__main__":
    main()
