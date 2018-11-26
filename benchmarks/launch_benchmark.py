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
from argparse import ArgumentParser
from common.base_benchmark_util import BaseBenchmarkUtil


class LaunchBenchmark(BaseBenchmarkUtil):
    """Launches benchmarking job based on the specified args """

    def main(self):
        super(LaunchBenchmark, self).define_args()

        arg_parser = ArgumentParser(parents=[self._common_arg_parser],
                                    description='Parse args for benchmark '
                                                'interface')

        # docker image
        arg_parser.add_argument("--docker-image",
                                help='Specify the docker image/tag to use',
                                dest='docker_image', default=None,
                                required=True)

        # checkpoint directory location
        arg_parser.add_argument('-c', "--checkpoint",
                                help='Specify the location of trained model '
                                     'checkpoint directory. '
                                     'If mode=training model/weights will be '
                                     'written to this location.'
                                     'If mode=inference assumes that the '
                                     'location points to a model that has '
                                     'already been trained. ',
                                dest='checkpoint', default=None)

        arg_parser.add_argument('-k', "--benchmark-only",
                                help='For benchmark measurement only.',
                                dest='benchmark_only',
                                action='store_true')

        arg_parser.add_argument("--accuracy-only",
                                help='For accuracy measurement only.',
                                dest='accuracy_only',
                                action='store_true')

        # in graph directory location
        arg_parser.add_argument('-g', "--in-graph",
                                help='Full path to the input graph ',
                                dest='input_graph', default=None)

        args, unknown = arg_parser.parse_known_args()
        self.validate_args(args)
        self.run_docker_container(args)

    def validate_args(self, args):
        """validate the args"""

        # validate the shared args first
        super(LaunchBenchmark, self).validate_args(args)

        # validate that we support this framework by checking folder names
        if glob.glob("*/{}".format(args.framework)) == []:
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

        # check if input graph file exists
        input_graph = args.input_graph
        if input_graph is not None:
            if not os.path.exists(input_graph):
                raise IOError("The input graph  {} does not exist.".
                              format(input_graph))
            if not os.path.isfile(input_graph):
                raise IOError("The input graph {} must be a file.".
                              format(input_graph))

        # check if benchmark-only and accuracy-only are all false for int8
        if args.platform == "int8" and (
                    not args.benchmark_only and not args.accuracy_only):
            raise ValueError("You must specify if the inference is for "
                             "benchmark or for accuracy measurement.")
        elif args.platform != "int8" and (
                    args.benchmark_only or args.accuracy_only):
            raise ValueError("{} and {} arguments are Int8 platform specific,"
                             " and not supported in {} inference.".
                             format(args.benchmark_only, args.accuracy_only,
                                    args.platform))

    def run_docker_container(self, args):
        """
        Runs a docker container with the specified image and environment
        variables to start running the benchmarking job.
        """
        benchmark_scripts = os.getcwd()
        intelai_models = os.path.join(benchmark_scripts, os.pardir, "models")

        if args.model_name:
            # find the path to the model's benchmarks folder
            search_path = os.path.join(
                benchmark_scripts, "*", args.framework, args.model_name,
                args.mode, args.platform)
            matches = glob.glob(search_path)
            if len(matches) > 1:
                # we should never get more than one match
                raise ValueError("Found multiple model locations for {} {} {}"
                                 .format(args.framework,
                                         args.model_name,
                                         args.platform))
            elif len(matches) == 0:
                raise ValueError("No model was found for {} {} {}"
                                 .format(args.framework,
                                         args.model_name,
                                         args.platform))

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

        env_vars = ("--env DATASET_LOCATION_VOL={} "
                    "--env CHECKPOINT_DIRECTORY_VOL={} "
                    "--env EXTERNAL_MODELS_SOURCE_DIRECTORY={} "
                    "--env INTELAI_MODELS={} "
                    "--env BENCHMARK_SCRIPTS={} "
                    "--env SINGLE_SOCKET={} "
                    "--env MODEL_NAME={} "
                    "--env MODE={} "
                    "--env PLATFORM={} "
                    "--env VERBOSE={} "
                    "--env BATCH_SIZE={} "
                    "--env WORKSPACE={} "
                    "--env IN_GRAPH=/in_graph/{} "
                    "--env MOUNT_BENCHMARK={} "
                    "--env MOUNT_EXTERNAL_MODELS_SOURCE={} "
                    "--env MOUNT_INTELAI_MODELS_SOURCE={} "
                    "--env USE_CASE={} "
                    "--env FRAMEWORK={} "
                    "--env NUM_CORES={} "
                    "--env DATASET_LOCATION=/dataset "
                    "--env CHECKPOINT_DIRECTORY=/checkpoints "
                    "--env BENCHMARK_ONLY={} "
                    "--env ACCURACY_ONLY={} "
                    .format(args.data_location, args.checkpoint,
                            args.model_source_dir, intelai_models,
                            benchmark_scripts, args.single_socket,
                            args.model_name, args.mode, args.platform,
                            args.verbose, args.batch_size, workspace,
                            in_graph_filename, mount_benchmark,
                            mount_external_models_source,
                            mount_intelai_models, use_case,
                            args.framework, args.num_cores,
                            args.benchmark_only, args.accuracy_only))

        # Add custom model args as env vars
        for custom_arg in args.model_args:
            if "=" not in custom_arg:
                raise ValueError("Expected model args in the format "
                                 "`name=value` but received: {}".
                                 format(custom_arg))

            env_vars = "{} --env {}".format(env_vars, custom_arg)

        volume_mounts = ("--volume {}:{} "
                         "--volume {}:{} "
                         "--volume {}:{} "
                         "--volume {}:/dataset "
                         "--volume {}:/checkpoints "
                         "--volume {}:/in_graph "
                         .format(benchmark_scripts, mount_benchmark,
                                 args.model_source_dir,
                                 mount_external_models_source,
                                 intelai_models, mount_intelai_models,
                                 args.data_location,
                                 args.checkpoint,
                                 in_graph_dir))

        docker_run_cmd = "docker run -it {} {} --privileged -u root:root " \
                         "-w {} {} /bin/bash start.sh"\
            .format(env_vars, volume_mounts, workspace,
                    args.docker_image)

        if args.verbose:
            print("Docker run command:\n{}".format(docker_run_cmd))

        # TODO: switch command to not be shell command
        p = subprocess.Popen(docker_run_cmd, shell=True)
        p.communicate()


if __name__ == "__main__":
    util = LaunchBenchmark()
    util.main()
