from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
from argparse import ArgumentParser
from common.base_benchmark_util import BaseBenchmarkUtil


class ModelBenchmarkUtil(BaseBenchmarkUtil):
    """Benchmark util for pf32 models """

    def main(self):
        super(ModelBenchmarkUtil, self).define_args()

        arg_parser = ArgumentParser(parents=[self._common_arg_parser],
                                    description='Parse args for benchmark interface')

        # checkpoint directory location
        arg_parser.add_argument('-c', "--checkpoint",
                                help='Specify the location of trained model checkpoint directory. '
                                     'If mode=training model/weights will be '
                                     'written to this location.'
                                     'If mode=inference assumes that the location '
                                     'points to a model that has already been trained. ',
                                dest='checkpoint', default=None)

        # in graph directory location
        arg_parser.add_argument('-g', "--in-graph",
                                help='Full path to the input graph ',
                                dest='input_graph', default=None)

        args, unknown = arg_parser.parse_known_args()
        self.validate_args(args)

        mi = super(ModelBenchmarkUtil, self).initialize_model(args, unknown)
        if mi is not None:  # start model initializer if not None
            mi.run()

    def validate_args(self, args):
        """validate the args for both common args and FP32 model specific args"""

        # validate the args shared by fp32 models and int8 models first
        super(ModelBenchmarkUtil, self).validate_args(args)

        # check checkpoint location
        checkpoint_dir = args.checkpoint
        if checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                raise IOError("The checkpoint location {} does not exist.".format(checkpoint_dir))
            elif not os.path.isdir(checkpoint_dir):
                raise IOError("The checkpoint location {} is not a directory.".format(checkpoint_dir))

        # check if input graph file exist
        input_graph = args.input_graph
        if input_graph is not None:
            if not os.path.exists(input_graph):
                raise IOError("The input graph  {} does not exist.".format(input_graph))
            if not os.path.isfile(input_graph):
                raise IOError("The input graph {} must be a file.".format(input_graph))


if __name__ == "__main__":
    util = ModelBenchmarkUtil()
    util.main()
