from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
from argparse import ArgumentParser
from common.platform_util import PlatformUtil


class BaseBenchmarkUtil(object):
    """Base benchmark util class for pf32 models and int8 models"""
    MODEL_INITIALIZER = "model_init"

    def __init__(self):
        self._common_arg_parser = None
        self._platform_util = PlatformUtil()

    def define_args(self):
        """define args for the benchmark interface shared by FP32 and int8 models"""

        DEFAULT_INTEROP_VALUE_ = self._platform_util.num_cpu_sockets()
        DEFAULT_INTRAOP_VALUE_ = self._platform_util.num_cores_per_socket() * self._platform_util.num_cpu_sockets()

        self._common_arg_parser = ArgumentParser(add_help=False, description='Parse args for base benchmark interface')

        self._common_arg_parser.add_argument('-f', "--framework",
                                             help='Specify the name of the '
                                                  'deep learning framework '
                                                  'to use.',
                                             dest='framework', default=None,
                                             required=True)

        self._common_arg_parser.add_argument('-r', "--model-source-dir",
                                             help="Specify the model source repositry to use " \
                                                  "from your local machine",
                                             nargs='?',
                                             dest="model_source_dir")

        self._common_arg_parser.add_argument('-p', "--platform",
                                             help="Specify the model platform to use " \
                                                  "fp32 or int8 or ats or bfloat16",
                                             required=True, choices=['fp32', 'int8', 'ats', 'bfloat16'],
                                             dest="platform")

        self._common_arg_parser.add_argument('-mo', "--mode",
                                             help="Specify the type training or inference ",
                                             required=True, choices=['training', 'inference'],
                                             dest="mode")

        self._common_arg_parser.add_argument('-m', "--model-name", required=True,
                                             help='model name to run benchmark for ',
                                             dest='model_name')

        self._common_arg_parser.add_argument('-b', "--batch-size",
                                             help='Specify the batch size. If this '
                                                  'parameter is not specified or is -1, the '
                                                  'largest ideal batch size for the model will '
                                                  'be used.',
                                             dest="batch_size", type=int, default=-1)

        self._common_arg_parser.add_argument('-d', "--data-location",
                                             help='Specify the location of the data. '
                                                  'If this parameter is not specified, '
                                                  'the benchmark will use random/dummy data.',
                                             dest="data_location", default=None)

        self._common_arg_parser.add_argument('-s', '--single-socket',
                                             help='Indicates that only one socket should '
                                                  'be used. If used in conjunction with '
                                                  '--num-cores, all cores will be allocated '
                                                  'on the single socket.',
                                             dest="single_socket", action='store_true')

        self._common_arg_parser.add_argument('-i', "--socket-id",
                                             help='Specify which socket to use. Default is socket 0.',
                                             dest="socket_id", type=int, default=0)

        self._common_arg_parser.add_argument('-n', "--num-cores",
                                             help='Specify the number of cores to use. '
                                                  'If the parameter is not specified '
                                                  'or is -1, all cores will be used.',
                                             dest="num_cores", type=int, default=-1)

        self._common_arg_parser.add_argument('-a', "--num_intra_threads", type=int,
                                             help="Specify the number of threads within the layer",
                                             dest="num_intra_threads",
                                             default=DEFAULT_INTRAOP_VALUE_)

        self._common_arg_parser.add_argument('-e', "--num_inter_threads", type=int,
                                             help='Specify the number threads between layers',
                                             dest="num_inter_threads",
                                             default=DEFAULT_INTEROP_VALUE_)

        self._common_arg_parser.add_argument('-v', "--verbose",
                                             help='Print verbose information.',
                                             dest='verbose',
                                             action='store_true')

    def validate_args(self, args):
        """validate the args """

        # check model_name exists
        if not args.model_name:
            raise ValueError("The model name {} is not valid".format(args.model_name))

        # check data location exist
        data_dir = args.data_location
        if data_dir is not None:
            if not os.path.exists(data_dir):
                raise IOError("The data location {} does not exist.".format(data_dir))

        # check batch size
        batch_size = args.batch_size
        if batch_size == 0 or batch_size < -1:
            raise ValueError("The batch size {} is not valid.".format(batch_size))

        # check if socket id is in socket number range
        num_sockets = self._platform_util.num_cpu_sockets()
        if (args.socket_id < 0) or (args.socket_id >= num_sockets):
            raise ValueError("Socket id must be within socket number range: [0, {}).".format(num_sockets))

        # check number of cores
        num_logical_cores_per_socket = self._platform_util.num_cores_per_socket() * self._platform_util.num_threads_per_core()
        system_num_cores = \
            num_logical_cores_per_socket if args.single_socket else num_logical_cores_per_socket * self._platform_util.num_cpu_sockets()
        num_cores = args.num_cores

        if (num_cores <= 0) and (num_cores != -1):
            raise ValueError(
                "Core number must be greater than 0 or -1. "
                "The default value is -1 which means using all the cores in the sockets")
        elif num_cores > system_num_cores:
            raise ValueError("Number of cores exceeds system core number: {}".format(system_num_cores))

    def initialize_model(self, args, unknown_args):
        """Create model initializer for the specified model"""

        model_initializer = None
        model_init_file = None
        if args.model_name:  # not empty
            current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

            # TODO: change this to search for the model_init.py based on the
            # framework, model, platform, and mode since we are going to be
            # adding a "use case" directory at some point, so we won't know the
            # exact location of the model_init.py file.
            filename = "{}.py".format(self.MODEL_INITIALIZER)
            print("current path: {}".format(current_path))
            search_path = os.path.join(current_path, "*", args.framework, args.model_name, args.mode, args.platform, filename)
            print("search path: {}".format(search_path))
            matches = glob.glob(search_path)
            
            if len(matches) > 1:
                # we should never get more than one match
                raise ValueError("Found multiple model_init.py files for {} {} {} {}".format(args.framework, args.model_name, args.platform, args.mode))
            elif len(matches) == 0:
                raise ValueError("No model_init.py was found for {} {} {} {}".format(args.framework, args.model_name, args.platform, args.mode))

            model_init_file = matches[0]

            print ("Using model init: {}".format(model_init_file))
            if os.path.exists(model_init_file):
                dir_list = model_init_file.split("/")
                framework_index = dir_list.index(args.framework)
                usecase = dir_list[framework_index - 1]                

                package = ".".join([usecase, args.framework, args.model_name, args.mode, args.platform])
                model_init_module = __import__(package + "."+self.MODEL_INITIALIZER, fromlist=['*'])
                model_initializer = model_init_module.ModelInitializer(args, unknown_args, self._platform_util)

        if model_initializer is None:
            raise ImportError("Unable to locate {}.".format(model_init_file))

        return model_initializer
