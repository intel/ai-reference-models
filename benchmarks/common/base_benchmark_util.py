#
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

import os

from argparse import ArgumentParser
from common import platform_util


class BaseBenchmarkUtil(object):
    """Base benchmark util class"""
    MODEL_INITIALIZER = "model_init"

    def __init__(self):
        self._common_arg_parser = None
        self._platform_util = platform_util.PlatformUtil()

    def define_args(self):
        """define args for the benchmark interface shared by FP32 and int8
        models"""

        self._common_arg_parser = ArgumentParser(
            add_help=False, description="Parse args for base benchmark "
                                        "interface")

        self._common_arg_parser.add_argument(
            "-f", "--framework",
            help="Specify the name of the deep learning framework to use.",
            dest="framework", default=None, required=True)

        self._common_arg_parser.add_argument(
            "-r", "--model-source-dir",
            help="Specify the models source directory from your local machine",
            nargs="?", dest="model_source_dir")

        self._common_arg_parser.add_argument(
            "-p", "--precision",
            help="Specify the model precision to use: fp32, int8, or bfloat16",
            required=True, choices=["fp32", "int8", "bfloat16"],
            dest="precision")

        self._common_arg_parser.add_argument(
            "-mo", "--mode", help="Specify the type training or inference ",
            required=True, choices=["training", "inference"], dest="mode")

        self._common_arg_parser.add_argument(
            "-m", "--model-name", required=True,
            help="model name to run benchmarks for", dest="model_name")

        self._common_arg_parser.add_argument(
            "-b", "--batch-size",
            help="Specify the batch size. If this parameter is not specified "
                 "or is -1, the largest ideal batch size for the model will "
                 "be used",
            dest="batch_size", type=int, default=-1)

        self._common_arg_parser.add_argument(
            "-d", "--data-location",
            help="Specify the location of the data. If this parameter is not "
                 "specified, the benchmark will use random/dummy data.",
            dest="data_location", default=None)

        self._common_arg_parser.add_argument(
            "-i", "--socket-id",
            help="Specify which socket to use. Only one socket will be used "
                 "when this value is set. If used in conjunction with "
                 "--num-cores, all cores will be allocated on the single "
                 "socket.",
            dest="socket_id", type=int, default=-1)

        self._common_arg_parser.add_argument(
            "-n", "--num-cores",
            help="Specify the number of cores to use. If the parameter is not"
                 " specified or is -1, all cores will be used.",
            dest="num_cores", type=int, default=-1)

        self._common_arg_parser.add_argument(
            "-a", "--num-intra-threads", type=int,
            help="Specify the number of threads within the layer",
            dest="num_intra_threads", default=None)

        self._common_arg_parser.add_argument(
            "-e", "--num-inter-threads", type=int,
            help="Specify the number threads between layers",
            dest="num_inter_threads", default=None)

        self._common_arg_parser.add_argument(
            "-c", "--checkpoint",
            help="Specify the location of trained model checkpoint directory. "
                 "If mode=training model/weights will be written to this "
                 "location. If mode=inference assumes that the location points"
                 " to a model that has already been trained.",
            dest="checkpoint", default=None)

        self._common_arg_parser.add_argument(
            "-g", "--in-graph", help="Full path to the input graph ",
            dest="input_graph", default=None)

        self._common_arg_parser.add_argument(
            "-k", "--benchmark-only",
            help="For benchmark measurement only. If neither --benchmark-only "
                 "or --accuracy-only are specified, it will default to run "
                 "benchmarking.",
            dest="benchmark_only", action="store_true")

        self._common_arg_parser.add_argument(
            "--accuracy-only",
            help="For accuracy measurement only.  If neither --benchmark-only "
                 "or --accuracy-only are specified, it will default to run "
                 "benchmarking.",
            dest="accuracy_only", action="store_true")

        self._common_arg_parser.add_argument(
            "--output-results",
            help="Writes inference output to a file, when used in conjunction "
                 "with --accuracy-only and --mode=inference.",
            dest="output_results", action="store_true")

        self._common_arg_parser.add_argument(
            "-v", "--verbose", help="Print verbose information.",
            dest="verbose", action="store_true")

        self._common_arg_parser.add_argument(
            "--output-dir",
            help="Folder to dump output into. The output directory will default to "
                 "'models/benchmarks/common/tensorflow/logs' if no path is specified.",
            default="/models/benchmarks/common/tensorflow/logs")

        # Allow for additional command line args after --
        self._common_arg_parser.add_argument(
            "model_args", nargs="*",
            help="Additional command line arguments (prefix flag start with"
                 " '--').")

    def check_for_link(self, arg_name, path):
        """
        Throws an error if the specified path is a link. os.islink returns
        True for sym links.  For files, we also look at the number of links in
        os.stat() to determine if it's a hard link.
        """
        if os.path.islink(path) or \
                (os.path.isfile(path) and os.stat(path).st_nlink > 1):
            raise ValueError("The {} cannot be a link.".format(arg_name))

    def validate_args(self, args):
        """validate the args """

        # check model source directory exists
        model_source_dir = args.model_source_dir
        if model_source_dir is not None:
            if not os.path.exists(model_source_dir) or \
                    not os.path.isdir(model_source_dir):
                raise IOError("The model source directory {} "
                              "does not exist or is not a directory.".
                              format(model_source_dir))
            self.check_for_link("model source directory", model_source_dir)

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

        # check model_name exists
        if not args.model_name:
            raise ValueError("The model name is not valid")

        # check batch size
        batch_size = args.batch_size
        if batch_size == 0 or batch_size < -1:
            raise ValueError("The batch size {} is not valid.".format(
                batch_size))

        # check data location exist
        data_dir = args.data_location
        if data_dir is not None:
            if not os.path.exists(data_dir):
                raise IOError("The data location {} does not exist.".format(
                    data_dir))
            self.check_for_link("data location", data_dir)

        # check if socket id is in socket number range
        num_sockets = self._platform_util.num_cpu_sockets()
        if args.socket_id != -1 and \
                (args.socket_id >= num_sockets or args.socket_id < -1):
            raise ValueError("Socket id must be within socket number range: "
                             "[0, {}].".format(num_sockets - 1))

        # check number of cores
        num_logical_cores_per_socket = \
            self._platform_util.num_cores_per_socket() * \
            self._platform_util.num_threads_per_core()
        # if a socket_id is specified, only count cores from one socket
        system_num_cores = num_logical_cores_per_socket if \
            args.socket_id != -1 else num_logical_cores_per_socket * \
            self._platform_util.num_cpu_sockets()
        num_cores = args.num_cores

        if (num_cores <= 0) and (num_cores != -1):
            raise ValueError(
                "Core number must be greater than 0 or -1. The default value "
                "is -1 which means using all the cores in the sockets")
        elif num_cores > system_num_cores:
            raise ValueError("Number of cores exceeds system core number: {}".
                             format(system_num_cores))

        # check no.of intra threads > 0
        num_intra_threads = args.num_intra_threads
        if num_intra_threads and num_intra_threads <= 0:
            raise ValueError("Number of intra threads "
                             "value should be greater than 0")

        # check no.of inter threads > 0
        num_inter_threads = args.num_inter_threads
        if num_inter_threads and num_inter_threads <= 0:
            raise ValueError("Number of inter threads "
                             "value should be greater than 0")

        if args.output_results and (args.mode != "inference" or not args.accuracy_only):
            raise ValueError("--output-results can only be used when running "
                             "with --mode=inference and --accuracy-only")
        elif args.output_results and (args.model_name != "resnet50" or args.precision != "fp32"):
            raise ValueError("--output-results is currently only supported for resnet50 FP32 inference.")

    def initialize_model(self, args, unknown_args):
        """Create model initializer for the specified model"""

        model_initializer = None
        model_init_file = None
        if args.model_name:  # not empty
            current_path = os.path.dirname(
                os.path.dirname(os.path.realpath(__file__)))

            # find the path to the model_init.py file
            filename = "{}.py".format(self.MODEL_INITIALIZER)
            model_init_file = os.path.join(current_path, args.use_case,
                                           args.framework, args.model_name,
                                           args.mode, args.precision,
                                           filename)
            package = ".".join([args.use_case, args.framework,
                                args.model_name, args.mode, args.precision])
            model_init_module = __import__(
                package + "." + self.MODEL_INITIALIZER, fromlist=["*"])
            model_initializer = model_init_module.ModelInitializer(
                args, unknown_args, self._platform_util)

        if model_initializer is None:
            raise ImportError("Unable to locate {}.".format(model_init_file))

        return model_initializer
