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

#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import signal
import subprocess
import sys
import platform as system_platform
from argparse import ArgumentParser
from common import base_benchmark_util
from common import platform_util
from common.utils.validators import check_no_spaces, check_volume_mount, check_shm_size
from common.base_model_init import BaseModelInitializer


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
        benchmark_scripts = os.path.dirname(os.path.realpath(__file__))
        os_type = system_platform.system()
        use_case = self.get_model_use_case(benchmark_scripts, os_type)
        intelai_models = self.get_model_dir(benchmark_scripts, use_case, os_type)
        intelai_models_common = self.get_model_dir(benchmark_scripts, "common", os_type)
        env_var_dict = self.get_env_vars(benchmark_scripts, use_case, intelai_models,
                                         intelai_models_common, os_type)
        if "Windows" == os_type:
            if os.getenv("PYTHONPATH") is None:
                os.environ["PYTHONPATH"] = os.path.dirname(sys.executable)
            os.environ["PYTHONPATH"] = "{};{};{}".format(
                benchmark_scripts, intelai_models, os.environ["PYTHONPATH"])

        if self.args.docker_image:
            if self.args.framework == 'tensorflow_serving':
                self.run_bare_metal(benchmark_scripts, intelai_models,
                                    intelai_models_common, env_var_dict, os_type)
            elif self.args.framework == 'tensorflow':
                self.run_docker_container(benchmark_scripts, intelai_models,
                                          intelai_models_common, env_var_dict)
        else:
            self.run_bare_metal(benchmark_scripts, intelai_models,
                                intelai_models_common, env_var_dict, os_type)

    def parse_args(self):
        # Additional args that are only used with the launch script
        arg_parser = ArgumentParser(
            parents=[self._common_arg_parser],
            description="Parse args for benchmark interface")

        arg_parser.add_argument(
            "--docker-image",
            help="Specify the docker image/tag to use when running benchmarking within a container."
                 "If no docker image is specified, then no docker container will be used.",
            dest="docker_image", default=None, type=check_no_spaces)

        arg_parser.add_argument(
            "--volume",
            help="Specify a custom volume to mount in the container, which follows the same format as the "
                 "docker --volume flag (https://docs.docker.com/storage/volumes/). "
                 "This argument can only be used in conjunction with a --docker-image.",
            action="append", dest="custom_volumes", type=check_volume_mount)

        arg_parser.add_argument(
            "--shm-size",
            help="Specify the size of docker /dev/shm. The format is <number><unit>. "
                 "number must be greater than 0. Unit is optional and can be b (bytes), k (kilobytes), "
                 "m (megabytes), or g (gigabytes).",
            dest="shm_size", default="64m", type=check_shm_size)

        arg_parser.add_argument(
            "--debug", help="Launches debug mode which doesn't execute "
                            "start.sh when running in a docker container.", action="store_true")

        arg_parser.add_argument(
            "--noinstall",
            help="whether to install packages for a given model when running in docker "
                 "(default --noinstall='False') or on bare metal (default --noinstall='True')",
            dest="noinstall", action="store_true", default=None)

        arg_parser.add_argument(
            "--dry-run",
            help="Shows the call to the model without actually running it",
            dest="dry_run", action="store_true", default=None)

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

        # default disable_tcmalloc=False for int8 and disable_tcmalloc=True for other precisions
        if not self.args.disable_tcmalloc:
            self.args.disable_tcmalloc = str(self.args.precision != "int8")

        if self.args.custom_volumes and not self.args.docker_image:
            raise ValueError("Volume mounts can only be used when running in a docker container "
                             "(a --docker-image must be specified when using --volume).")

        if self.args.mode == "inference" and self.args.checkpoint:
            print("Warning: The --checkpoint argument is being deprecated in favor of using frozen graphs.")

    def get_model_use_case(self, benchmark_scripts, os_type):
        """
        Infers the use case based on the directory structure for the specified model.
        """
        args = self.args

        # find the path to the model's benchmarks folder
        search_path = os.path.join(
            benchmark_scripts, "*", args.framework, args.model_name,
            args.mode, args.precision)
        matches = glob.glob(search_path)
        error_str = ""
        if len(matches) > 1:
            error_str = "Found multiple model locations for {} {} {}"
        elif len(matches) == 0:
            error_str = "No model was found for {} {} {}"
        if error_str:
            raise ValueError(error_str.format(args.framework, args.model_name, args.precision))

        # use the benchmarks directory path to find the use case
        if "Windows" == os_type:
            dir_list = matches[0].split("\\")
        else:
            dir_list = matches[0].split("/")

        # find the last occurrence of framework in the list, then return
        # the element before it in the path, which is the use case
        return next(dir_list[elem - 1] for elem in range(len(dir_list) - 1, -1, -1)
                    if dir_list[elem] == args.framework)

    def get_model_dir(self, benchmark_scripts, use_case, os_type):
        """
        Finds the path to the optimized model directory in this repo, if it exists.
        """

        # use the models directory as a default
        intelai_models = os.path.join(benchmark_scripts, os.pardir, "models")

        if use_case == "common":
            return os.path.join(intelai_models, "common", self.args.framework)

        # find the intelai_optimized model directory
        args = self.args
        optimized_model_dir = os.path.join(intelai_models, use_case,
                                           args.framework, args.model_name)
        if "Windows" == os_type:
            optimized_model_dir = optimized_model_dir.replace("\\", "/")

        # if we find an optimized model, then we will use that path
        if os.path.isdir(optimized_model_dir):
            intelai_models = optimized_model_dir
        return intelai_models

    def get_env_vars(self, benchmark_scripts, use_case, intelai_models,
                     intelai_models_common, os_type):
        """
        Sets up dictionary of standard env vars that are used by start.sh
        """
        # Standard env vars
        args = self.args
        python_exe = str(sys.executable if not args.docker_image else "python")
        if "Windows" == os_type:
            python_exe = r'"{}"'.format(python_exe)

        env_var_dict = {
            "ACCURACY_ONLY": args.accuracy_only,
            "BACKBONE_MODEL_DIRECTORY_VOL": args.backbone_model,
            "BATCH_SIZE": args.batch_size,
            "BENCHMARK_ONLY": args.benchmark_only,
            "BENCHMARK_SCRIPTS": benchmark_scripts,
            "CHECKPOINT_DIRECTORY_VOL": args.checkpoint,
            "DATASET_LOCATION_VOL": args.data_location,
            "DATA_NUM_INTER_THREADS": args.data_num_inter_threads,
            "DATA_NUM_INTRA_THREADS": args.data_num_intra_threads,
            "DISABLE_TCMALLOC": args.disable_tcmalloc,
            "DOCKER": str(args.docker_image) if args.docker_image is not None else "",
            "DRY_RUN": str(args.dry_run) if args.dry_run is not None else "",
            "EXTERNAL_MODELS_SOURCE_DIRECTORY": args.model_source_dir,
            "FRAMEWORK": args.framework,
            "INTELAI_MODELS": intelai_models,
            "INTELAI_MODELS_COMMON": intelai_models_common,
            "MODE": args.mode,
            "MODEL_NAME": args.model_name,
            "MPI_HOSTNAMES": args.mpi_hostnames,
            "MPI_NUM_PROCESSES": args.mpi,
            "MPI_NUM_PROCESSES_PER_SOCKET": args.num_mpi,
            "NUMA_CORES_PER_INSTANCE": args.numa_cores_per_instance,
            "NOINSTALL": str(args.noinstall) if args.noinstall is not None else "True" if not args.docker_image else "False",  # noqa: E501
            "NUM_CORES": args.num_cores,
            "NUM_INTER_THREADS": args.num_inter_threads,
            "NUM_INTRA_THREADS": args.num_intra_threads,
            "NUM_TRAIN_STEPS": args.num_train_steps,
            "OUTPUT_RESULTS": args.output_results,
            "PRECISION": args.precision,
            "PYTHON_EXE": python_exe,
            "SOCKET_ID": args.socket_id,
            "TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD": args.tcmalloc_large_alloc_report_threshold,
            "TF_SERVING_VERSION": args.tf_serving_version,
            "USE_CASE": str(use_case),
            "VERBOSE": args.verbose,
            "WEIGHT_SHARING": args.weight_sharing
        }

        # Add custom model args as env vars)
        for custom_arg in args.model_args + self.unknown_args:
            if "=" not in custom_arg:
                raise ValueError("Expected model args in the format "
                                 "`name=value` but received: {}".
                                 format(custom_arg))
            split_arg = custom_arg.split("=")
            split_arg[0] = split_arg[0].replace("-", "_").lstrip('_')
            # Convert the arg names to upper case to work on both Windows and Linux systems
            split_arg[0] = split_arg[0].upper()
            env_var_dict[split_arg[0]] = split_arg[1]

        return env_var_dict

    def run_bare_metal(self, benchmark_scripts, intelai_models,
                       intelai_models_common, env_var_dict, os_type):
        """
        Runs the model without a container
        """
        # setup volume directories to be the local system directories, since we aren't
        # mounting volumes when running bare metal, but start.sh expects these args
        args = self.args
        workspace = os.path.join(benchmark_scripts, "common", args.framework)
        mount_benchmark = benchmark_scripts
        in_graph_path = args.input_graph
        checkpoint_path = args.checkpoint
        backbone_model_path = args.backbone_model
        dataset_path = args.data_location

        mount_external_models_source = args.model_source_dir
        mount_intelai_models = intelai_models

        # To Launch Tensorflow Serving benchmark we need only --in-graph arg.
        # It does not support checkpoint files.
        if args.framework == "tensorflow_serving":
            if checkpoint_path:
                raise ValueError("--checkpoint-path arg is not supported with tensorflow serving benchmarking")

            if args.mode != "inference":
                raise ValueError("--mode arg should be set to inference")

            if in_graph_path:
                env_var_dict["IN_GRAPH"] = in_graph_path
            else:
                raise ValueError("--in-graph arg is required to run tensorflow serving benchmarking")

            for env_var_name in env_var_dict:
                os.environ[env_var_name] = str(env_var_dict[env_var_name])

            # We need this env to be set for the platform util
            os.environ["PYTHON_EXE"] = str(sys.executable if not args.docker_image else "python")
            # Get Platformutil
            platform_util_obj = None or platform_util.PlatformUtil(self.args)
            # Configure num_inter_threads and num_intra_threads
            base_obj = BaseModelInitializer(args=self.args, custom_args=[], platform_util=platform_util_obj)
            base_obj.set_num_inter_intra_threads()

            # Update num_inter_threads and num_intra_threads in env dictionary
            env_var_dict["NUM_INTER_THREADS"] = self.args.num_inter_threads
            env_var_dict["NUM_INTRA_THREADS"] = self.args.num_intra_threads

            # Set OMP_NUM_THREADS
            env_var_dict["OMP_NUM_THREADS"] = self.args.num_intra_threads

        else:
            mount_external_models_source = args.model_source_dir
            mount_intelai_models = intelai_models
            mount_intelai_models_common = intelai_models_common

            # Add env vars with bare metal settings
            env_var_dict["MOUNT_EXTERNAL_MODELS_SOURCE"] = mount_external_models_source
            env_var_dict["MOUNT_INTELAI_MODELS_SOURCE"] = mount_intelai_models
            env_var_dict["MOUNT_INTELAI_MODELS_COMMON_SOURCE"] = mount_intelai_models_common

            if in_graph_path:
                env_var_dict["IN_GRAPH"] = in_graph_path

            if checkpoint_path:
                env_var_dict["CHECKPOINT_DIRECTORY"] = checkpoint_path

            if backbone_model_path:
                env_var_dict["BACKBONE_MODEL_DIRECTORY"] = backbone_model_path

        if dataset_path:
            env_var_dict["DATASET_LOCATION"] = dataset_path

        # if using the default output directory, get the full path
        if args.output_dir == "/models/benchmarks/common/tensorflow/logs":
            args.output_dir = os.path.join(workspace, "logs")

        # Add env vars with bare metal settings
        env_var_dict["WORKSPACE"] = workspace
        env_var_dict["MOUNT_BENCHMARK"] = mount_benchmark
        env_var_dict["OUTPUT_DIR"] = args.output_dir

        # Set env vars for bare metal
        for env_var_name in env_var_dict:
            os.environ[env_var_name] = str(env_var_dict[env_var_name])

        # Run the start script
        start_script = os.path.join(workspace, "start.sh")
        if "Windows" == os_type:
            self._launch_command([os.environ["MSYS64_BASH"], start_script])
        else:
            self._launch_command(["bash", start_script])

    def run_docker_container(self, benchmark_scripts, intelai_models,
                             intelai_models_common, env_var_dict):
        """
        Runs a docker container with the specified image and environment
        variables to start running the benchmarking job.
        """
        args = self.args
        mount_benchmark = "/workspace/benchmarks"
        mount_external_models_source = "/workspace/models"
        mount_intelai_models = "/workspace/intelai_models"
        mount_intelai_models_common = "/workspace/intelai_models_common"
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

        # env vars with docker settings
        env_vars = ["--env", "WORKSPACE={}".format(workspace),
                    "--env", "MOUNT_BENCHMARK={}".format(mount_benchmark),
                    "--env", "MOUNT_EXTERNAL_MODELS_SOURCE={}".format(mount_external_models_source),
                    "--env", "MOUNT_INTELAI_MODELS_SOURCE={}".format(mount_intelai_models),
                    "--env", "MOUNT_INTELAI_MODELS_COMMON_SOURCE={}".format(mount_intelai_models_common),
                    "--env", "OUTPUT_DIR={}".format(output_dir)]

        if args.input_graph:
            env_vars += ["--env", "IN_GRAPH=/in_graph/{}".format(in_graph_filename)]

        if args.data_location:
            env_vars += ["--env", "DATASET_LOCATION=/dataset"]

        if args.checkpoint:
            env_vars += ["--env", "CHECKPOINT_DIRECTORY=/checkpoints"]

        if args.backbone_model:
            env_vars += ["--env", "BACKBONE_MODEL_DIRECTORY=/backbone_model"]

        # Add env vars with common settings
        for env_var_name in env_var_dict:
            env_vars += ["--env", "{}={}".format(env_var_name, env_var_dict[env_var_name])]

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
                         "--volume", "{}:{}".format(intelai_models_common, mount_intelai_models_common)]

        if mount_output_dir:
            volume_mounts.extend([
                "--volume", "{}:{}".format(output_dir, output_dir)])

        if args.data_location:
            volume_mounts.extend([
                "--volume", "{}:{}".format(args.data_location, "/dataset")])

        if args.checkpoint:
            volume_mounts.extend([
                "--volume", "{}:{}".format(args.checkpoint, "/checkpoints")])

        if args.backbone_model:
            volume_mounts.extend([
                "--volume", "{}:{}".format(args.backbone_model, "/backbone_model")])

        if in_graph_dir:
            volume_mounts.extend([
                "--volume", "{}:{}".format(in_graph_dir, "/in_graph")])

        if args.custom_volumes:
            for custom_volume in args.custom_volumes:
                volume_mounts.extend(["--volume", custom_volume])

        docker_run_cmd = ["docker", "run"]

        # only use -it when debugging, otherwise we might get TTY error
        if args.debug:
            docker_run_cmd.append("-it")

        if args.numa_cores_per_instance is not None or args.socket_id != -1 or \
                args.num_cores != -1 or args.mpi is not None or args.num_mpi > 1:
            docker_run_cmd.append("--privileged")

        docker_shm_size = "--shm-size={}".format(args.shm_size)
        docker_run_cmd = docker_run_cmd + env_vars + volume_mounts + [
            docker_shm_size, "-u", "root:root", "-w",
            workspace, args.docker_image, "/bin/bash"]

        if not args.debug:
            docker_run_cmd.append("start.sh")

        if args.verbose:
            print("Docker run command:\n{}".format(docker_run_cmd))

        self._launch_command(docker_run_cmd)

    def _launch_command(self, run_cmd):
        """runs command that runs the start script in a container or on bare metal and exits on ctrl c"""
        os_type = system_platform.system()
        if "Windows" == os_type:
            p = subprocess.Popen(run_cmd, start_new_session=True)
        else:
            p = subprocess.Popen(run_cmd, preexec_fn=os.setsid)

        try:
            p.communicate()
        except KeyboardInterrupt:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)


if __name__ == "__main__":
    util = LaunchBenchmark()
    util.main()
