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
import subprocess
import sys
from argparse import ArgumentParser
from common import base_benchmark_util


class LaunchBenchmark(base_benchmark_util.BaseBenchmarkUtil):
    """Launches benchmarking job based on the specified args """

    def main(self):
        args, unknown = self.parse_args(sys.argv[1:])
        try:
            self.validate_args(args)
        except (IOError, ValueError) as e:
            print("\nError: {}".format(e))
            sys.exit(1)
        self.run_docker_container(args)

    def parse_args(self, args):
        super(LaunchBenchmark, self).define_args()

        arg_parser = ArgumentParser(
            parents=[self._common_arg_parser],
            description="Parse args for benchmark interface")

        # docker image
        arg_parser.add_argument(
            "--docker-image", help="Specify the docker image/tag to use",
            dest="docker_image", default=None, required=True)

        # checkpoint directory location
        arg_parser.add_argument(
            "-c", "--checkpoint",
            help="Specify the location of trained model checkpoint directory. "
                 "If mode=training model/weights will be written to this "
                 "location. If mode=inference assumes that the location points"
                 " to a model that has already been trained.",
            dest="checkpoint", default=None)

        arg_parser.add_argument(
            "-k", "--benchmark-only",
            help="For benchmark measurement only. If neither --benchmark-only "
                 "or --accuracy-only are specified, it will default to run "
                 "benchmarking.",
            dest="benchmark_only", action="store_true")

        arg_parser.add_argument(
            "--accuracy-only",
            help="For accuracy measurement only.  If neither --benchmark-only "
                 "or --accuracy-only are specified, it will default to run "
                 "benchmarking.",
            dest="accuracy_only", action="store_true")

        # in graph directory location
        arg_parser.add_argument(
            "-g", "--in-graph", help="Full path to the input graph ",
            dest="input_graph", default=None)

        arg_parser.add_argument(
            "--debug", help="Launches debug mode which doesn't execute "
            "start.sh", action="store_true")

        return arg_parser.parse_known_args(args)

    def validate_args(self, args):
        """validate the args"""

        # validate the shared args first
        super(LaunchBenchmark, self).validate_args(args)

        # Check for spaces in docker image
        if ' ' in args.docker_image:
            raise ValueError("docker image string "
                             "should not have whitespace(s)")

        # validate that we support this framework by checking folder names
        benchmark_dir = os.path.dirname(os.path.realpath(__file__))
        if glob.glob("{}/*/{}".format(benchmark_dir, args.framework)) == []:
            raise ValueError("The specified framework is not supported: {}".
                             format(args.framework))

        # check checkpoint location
        checkpoint_dir = args.checkpoint
        if checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                raise IOError("The checkpoint location {} does not exist.".
                              format(checkpoint_dir))
            elif not os.path.isdir(checkpoint_dir):
                raise IOError("The checkpoint location {} is not a directory.".
                              format(checkpoint_dir))
            self.check_for_link("checkpoint directory", checkpoint_dir)

        # check if input graph file exists
        input_graph = args.input_graph
        if input_graph is not None:
            if not os.path.exists(input_graph):
                raise IOError("The input graph {} does not exist.".
                              format(input_graph))
            if not os.path.isfile(input_graph):
                raise IOError("The input graph {} must be a file.".
                              format(input_graph))
            self.check_for_link("input graph", input_graph)

        # if neither benchmark_only or accuracy_only are specified, then enable
        # benchmark_only as the default
        if not args.benchmark_only and not args.accuracy_only:
            args.benchmark_only = True

    def run_docker_container(self, args):
        """
        Runs a docker container with the specified image and environment
        variables to start running the benchmarking job.
        """
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
                    "--env", "DATASET_LOCATION=/dataset",
                    "--env", "CHECKPOINT_DIRECTORY=/checkpoints",
                    "--env", "BENCHMARK_ONLY={}".format(args.benchmark_only),
                    "--env", "ACCURACY_ONLY={}".format(args.accuracy_only),
                    "--env", "NOINSTALL=False"]

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

        p = subprocess.Popen(docker_run_cmd)
        p.communicate()


if __name__ == "__main__":
    util = LaunchBenchmark()
    util.main()
