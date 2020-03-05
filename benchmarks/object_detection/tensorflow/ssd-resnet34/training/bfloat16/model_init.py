#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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

import os
import sys
import argparse

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


def run_training_checks(args):
    if not args.data_location:
        sys.exit("Please provide a path to the data directory via the '--data-location' flag.")


class ModelInitializer(BaseModelInitializer):

    def __init__(self, args, custom_args, platform_util):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        run_training_checks(self.args)
        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        self.set_num_inter_intra_threads()

        # Train parameter parser
        parser = argparse.ArgumentParser(description="process custom_args")
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--num_warmup_batches', type=int, default=20)
        parser.add_argument('--num_train_steps', type=int, default=500, help='number of training batches')
        parser.add_argument('--num_processes', type=int, default=2, help='number of process')
        parser.add_argument('--num_processes_per_node', type=int, default=1, help='number of process per node')
        parser.add_argument('--num_inter_threads', type=int, default=1, help='number of inter-threads')
        parser.add_argument('--num_intra_threads', type=int, default=28, help='number of intra-threads')
        # parser.add_argument('--enable_bfloat16', action='store_true',
        #                     help='If True, rewrite graph to use bfloat16')
        self.args = parser.parse_args(self.custom_args, namespace=self.args)

        omp_num_threads = platform_util.num_cores_per_socket

        # set_env_var("OMP_NUM_THREADS", omp_num_threads if self.args.num_cores == -1 else self.args.num_cores)
        set_env_var("OMP_NUM_THREADS", 28)

        cmd_args = " --data_dir {0}".format(self.args.data_location)
        cmd_args += " --batch_size {0}".format(self.args.batch_size)
        cmd_args += " --weight_decay {0}".format(self.args.weight_decay)
        cmd_args += " --num_warmup_batches {0}".format(self.args.num_warmup_batches)
        cmd_args += " --num_batches {0}".format(self.args.num_train_steps)
        cmd_args += " --num_inter_threads {0}".format(self.args.num_inter_threads)
        cmd_args += " --num_intra_threads {0}".format(self.args.num_intra_threads)
        cmd_args += " --model=ssd300 --data_name coco"
        cmd_args += " --mkl=True --device=cpu --data_format=NCHW"
        cmd_args += " --rewriter_config=convert_to_bfloat16:ON"
        # cmd_args += " --variable_update=horovod --horovod_device=cpu"
        # cmd_args += " --trace_file=ssd-rn34-trace-bfloat16-no-mpi.json --use_chrome_trace_format=True"

        multi_instance_param_list = ["-genv:I_MPI_PIN_DOMAIN=socket",
                                     "-genv:I_MPI_FABRICS=shm",
                                     "-genv:I_MPI_ASYNC_PROGRESS=1",
                                     "-genv:I_MPI_ASYNC_PROGRESS_PIN={},{}".format(0, self.args.num_intra_threads),
                                     "-genv:OMP_NUM_THREADS={}".format(self.args.num_intra_threads)]
        # self.cmd = self.get_multi_instance_train_prefix(multi_instance_param_list)
        # self.cmd += "{} ".format(self.python_exe)
        self.cmd = "numactl --cpunodebind=0 --membind=0 {} ".format(self.python_exe)

        self.training_script_dir = os.path.join('/tmp/benchmark_ssd-resnet34/scripts/tf_cnn_benchmarks')
        training_script = os.path.join(self.training_script_dir, 'tf_cnn_benchmarks.py')

        self.cmd = self.cmd + training_script + cmd_args
        print("Ran command: {}".format(self.cmd))

    def run(self):
        original_dir = os.getcwd()
        os.chdir(self.training_script_dir)
        # Run benchmarking
        self.run_command(self.cmd)
        os.chdir(original_dir)
