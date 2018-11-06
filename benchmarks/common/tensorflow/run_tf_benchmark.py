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
        mi = super(ModelBenchmarkUtil, self).initialize_model(args, unknown)
        if mi is not None:  # start model initializer if not None
            mi.run()

if __name__ == "__main__":
    util = ModelBenchmarkUtil()
    util.main()
