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

        DEFAULT_INTEROP_VALUE_ = self._platform_util.num_cpu_sockets()
        DEFAULT_INTRAOP_VALUE_ = self._platform_util.num_cores_per_socket() * \
            self._platform_util.num_cpu_sockets()

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
            dest="num_intra_threads", default=DEFAULT_INTRAOP_VALUE_)

        self._common_arg_parser.add_argument(
            "-e", "--num-inter-threads", type=int,
            help="Specify the number threads between layers",
            dest="num_inter_threads", default=DEFAULT_INTEROP_VALUE_)

        self._common_arg_parser.add_argument(
            "-v", "--verbose", help="Print verbose information.",
            dest="verbose", action="store_true")

        # Allow for additional command line args after --
        self._common_arg_parser.add_argument(
            "model_args", nargs="*",
            help="Additional command line arguments (prefix flag start with"
                 " '--').")

    def validate_args(self, args):
        """validate the args """

        # check model_name exists
        if not args.model_name:
            raise ValueError("The model name is not valid")

        # check data location exist
        data_dir = args.data_location
        if data_dir is not None:
            if not os.path.exists(data_dir):
                raise IOError("The data location {} does not exist.".format(
                    data_dir))

        # check batch size
        batch_size = args.batch_size
        if batch_size == 0 or batch_size < -1:
            raise ValueError("The batch size {} is not valid.".format(
                batch_size))

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
            model_init_module = __import__(package + "." +
                                           self.MODEL_INITIALIZER,
                                           fromlist=["*"])
            model_initializer = model_init_module.ModelInitializer(
                args, unknown_args, self._platform_util)

        if model_initializer is None:
            raise ImportError("Unable to locate {}.".format(model_init_file))

        return model_initializer
