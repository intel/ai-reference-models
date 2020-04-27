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

#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
from common.base_benchmark_util import BaseBenchmarkUtil


class ModelBenchmarkUtil(BaseBenchmarkUtil):
    """Benchmark util for int8 and fp32 models """

    def main(self):
        # Additional args that are used internally for the start script to call the model_init.py
        arg_parser = ArgumentParser(parents=[self._common_arg_parser],
                                    description='Parse args for benchmark '
                                                'interface')

        arg_parser.add_argument("--intelai-models",
                                help="Local path to the intelai optimized "
                                     "model scripts",
                                nargs='?',
                                dest="intelai_models")

        arg_parser.add_argument("--benchmark-dir",
                                help="Local path intelai benchmark directory",
                                nargs='?',
                                dest="benchmark_dir")

        arg_parser.add_argument("--use-case",
                                help="The corresponding use case of the given "
                                     "model ",
                                nargs='?',
                                dest="use_case")

        args, unknown = arg_parser.parse_known_args()
        mi = self.initialize_model(args, unknown)
        if mi is not None:  # start model initializer if not None
            mi.run()


if __name__ == "__main__":
    util = ModelBenchmarkUtil()
    util.main()
