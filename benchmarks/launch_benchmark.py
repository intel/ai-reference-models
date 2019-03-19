#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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
# SPDX-License-Identifier: EPL-2.0
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import signal
import subprocess
import sys
from argparse import ArgumentParser
from common import base_benchmark_util
from common.utils.validators import check_no_spaces


class LaunchBenchmark(base_benchmark_util.BaseBenchmarkUtil):
    """Launches benchmarking job based on the specified args """

    def __init__(self, *args, **kwargs):
        super(LaunchBenchmark, self).__init__(*args, **kwargs)

        self.args, self.unknown_args = self.parse_args()
        try:
            self.validate_args()
        except (IOError, ValueError) as e:
            sys.exit("\nError: {}".format(e))

    def main(self):
        self.run_docker_container()

    def parse_args(self):
        # Additional args that are only used with the launch script
        arg_parser = ArgumentParser(
            parents=[self._common_arg_parser],
            description="Parse args for benchmark interface")

        arg_parser.add_argument(
            "--docker-image", help="Specify the docker image/tag to use",
            dest="docker_image", default=None, required=True, type=check_no_spaces)

        arg_parser.add_argument(
            "--debug", help="Launches debug mode which doesn't execute "
            "start.sh", action="store_true")

        return arg_parser.parse_known_args()

    def validate_args(self):
        """validate the args"""
        # validate that we support this framework by checking folder names
        benchmark_dir = os.path.dirname(os.path.realpath(__file__))
        if glob.glob("{}/*/{}".format(benchmark_dir, self.args.framework)) == []:
            raise ValueError("The specified framework is not supported: {}".
                             format(self.args.framework))

        # if neither benchmark_only or accuracy_only are specified, then enable
        # benchmark_only as the default
        if not self.args.benchmark_only and not self.args.accuracy_only:
            self.args.benchmark_only = True

    def run_docker_container(self):
        """
        Runs a docker container with the specified image and environment
        variables to start running the benchmarking job.
        """
        args = self.args
        benchmark_scripts = os.path.dirname(os.path.realpath(__file__))
        intelai_models = os.path.join(benchmark_scripts, os.pardir, "models")

        if args.model_name:
            # find the path to the model's benchmarks folder
            search_path = os.path.join(
                benchmark_scripts, "*", args.framework, args.model_name,
                args.mode, args.precision)
            matches = glob.glob(search_path)
            if len(matches) > 1:
                # we should never get more than one match
                raise ValueError("Found multiple model locations for {} {} {}"
                                 .format(args.framework,
                                         args.model_name,
                                         args.precision))
            elif len(matches) == 0:
                raise ValueError("No model was found for {} {} {}"
                                 .format(args.framework,
                                         args.model_name,
                                         args.precision))

            # use the benchmarks directory path to find the use case
            dir_list = matches[0].split("/")

            # find the last occurrence of framework in the list
            framework_index = len(dir_list) - 1 - dir_list[::-1].index(
                args.framework)

            # grab the use case name from the path
            use_case = str(dir_list[framework_index - 1])

            # find the intelai_optimized model directory
            optimized_model_dir = os.path.join(
                benchmark_scripts, os.pardir, "models", use_case,
                args.framework, args.model_name)

            # if we find an optimized model, then we will use that path
            if os.path.isdir(intelai_models):
                intelai_models = optimized_model_dir

        mount_benchmark = "/workspace/benchmarks"
        mount_external_models_source = "/workspace/models"
        mount_intelai_models = "/workspace/intelai_models"
        workspace = os.path.join(mount_benchmark, "common", args.framework)

        mount_output_dir = False
        output_dir = os.path.join(workspace, 'logs')
        if args.output_dir != "/models/benchmarks/common/tensorflow/logs":
            # we don't need to mount log dir otherwise since default is workspace folder
            mount_output_dir = True
            output_dir = args.output_dir

        in_graph_dir = os.path.dirname(args.input_graph) if args.input_graph \
            else ""
        in_graph_filename = os.path.basename(args.input_graph) if \
            args.input_graph else ""

        env_vars = ["--env", "DATASET_LOCATION_VOL={}".format(args.data_location),
                    "--env", "CHECKPOINT_DIRECTORY_VOL={}".format(args.checkpoint),
                    "--env", "EXTERNAL_MODELS_SOURCE_DIRECTORY={}".format(args.model_source_dir),
                    "--env", "INTELAI_MODELS={}".format(intelai_models),
                    "--env", "BENCHMARK_SCRIPTS={}".format(benchmark_scripts),
                    "--env", "SOCKET_ID={}".format(args.socket_id),
                    "--env", "MODEL_NAME={}".format(args.model_name),
                    "--env", "MODE={}".format(args.mode),
                    "--env", "PRECISION={}".format(args.precision),
                    "--env", "VERBOSE={}".format(args.verbose),
                    "--env", "BATCH_SIZE={}".format(args.batch_size),
                    "--env", "WORKSPACE={}".format(workspace),
                    "--env", "IN_GRAPH=/in_graph/{}".format(in_graph_filename),
                    "--env", "MOUNT_BENCHMARK={}".format(mount_benchmark),
                    "--env", "MOUNT_EXTERNAL_MODELS_SOURCE={}".format(mount_external_models_source),
                    "--env", "MOUNT_INTELAI_MODELS_SOURCE={}".format(mount_intelai_models),
                    "--env", "USE_CASE={}".format(use_case),
                    "--env", "FRAMEWORK={}".format(args.framework),
                    "--env", "NUM_CORES={}".format(args.num_cores),
                    "--env", "NUM_INTER_THREADS={}".format(args.num_inter_threads),
                    "--env", "NUM_INTRA_THREADS={}".format(args.num_intra_threads),
                    "--env", "DATA_NUM_INTER_THREADS={}".format(args.data_num_inter_threads),
                    "--env", "DATA_NUM_INTRA_THREADS={}".format(args.data_num_intra_threads),
                    "--env", "DATASET_LOCATION=/dataset",
                    "--env", "CHECKPOINT_DIRECTORY=/checkpoints",
                    "--env", "BENCHMARK_ONLY={}".format(args.benchmark_only),
                    "--env", "ACCURACY_ONLY={}".format(args.accuracy_only),
                    "--env", "OUTPUT_RESULTS={}".format(args.output_results),
                    "--env", "NOINSTALL=False",
                    "--env", "OUTPUT_DIR={}".format(output_dir)]

        # by default we will install, user needs to set NOINSTALL=True
        # manually after they get into `--debug` mode
        # since they need to run one time without this flag
        # to get stuff installed

        # Add custom model args as env vars
        for custom_arg in args.model_args:
            if "=" not in custom_arg:
                raise ValueError("Expected model args in the format "
                                 "`name=value` but received: {}".
                                 format(custom_arg))
            env_vars.append("--env")
            env_vars.append("{}".format(custom_arg))

        # Add proxy to env variables if any set on host
        for environment_proxy_setting in [
            "http_proxy",
            "ftp_proxy",
            "https_proxy",
            "no_proxy",
        ]:
            if not os.environ.get(environment_proxy_setting):
                continue
            env_vars.append("--env")
            env_vars.append("{}={}".format(
                environment_proxy_setting,
                os.environ.get(environment_proxy_setting)
            ))

        volume_mounts = ["--volume", "{}:{}".format(benchmark_scripts, mount_benchmark),
                         "--volume", "{}:{}".format(args.model_source_dir, mount_external_models_source),
                         "--volume", "{}:{}".format(intelai_models, mount_intelai_models),
                         "--volume", "{}:/dataset".format(args.data_location),
                         "--volume", "{}:/checkpoints".format(args.checkpoint),
                         "--volume", "{}:/in_graph".format(in_graph_dir)]

        if mount_output_dir:
            volume_mounts.extend([
                "--volume", "{}:{}".format(output_dir, output_dir)])

        docker_run_cmd = ["docker", "run"]

        # only use -it when debugging, otherwise we might get TTY error
        if args.debug:
            docker_run_cmd.append("-it")

        docker_run_cmd = docker_run_cmd + env_vars + volume_mounts + [
            "--privileged", "-u", "root:root", "-w",
            workspace, args.docker_image, "/bin/bash"]

        if not args.debug:
            docker_run_cmd.append("start.sh")

        if args.verbose:
            print("Docker run command:\n{}".format(docker_run_cmd))

        self._run_docker_cmd(docker_run_cmd)

    def _run_docker_cmd(self, docker_run_cmd):
        """runs docker proc and exits on ctrl c"""
        p = subprocess.Popen(docker_run_cmd, preexec_fn=os.setsid)
        try:
            p.communicate()
        except KeyboardInterrupt:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)


if __name__ == "__main__":
    util = LaunchBenchmark()
    util.main()
