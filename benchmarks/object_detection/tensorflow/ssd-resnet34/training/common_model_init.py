#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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
import time

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


def run_training_checks(args):
    if not args.data_location:
        sys.exit("Please provide a path to the data directory via the '--data-location' flag.")


class SSDResnet34ModelInitializer(BaseModelInitializer):

    def __init__(self, args, custom_args, platform_util):
        super(SSDResnet34ModelInitializer, self).__init__(args, custom_args, platform_util)

        run_training_checks(self.args)
        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        # Train parameter parser
        parser = argparse.ArgumentParser(description="process custom_args")
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--num_warmup_batches', type=int, default=0)
        parser.add_argument('--num_train_steps', type=int, default=500, help='number of training batches')
        parser.add_argument('--num_inter_threads', type=int, default=1, help='number of inter-threads')
        parser.add_argument('--num_intra_threads', type=int, default=-1, help='number of intra-threads')
        parser.add_argument('--epochs', dest="epochs", type=int, default=60,
                            help='number of training epochs. Pass 0 to train based on number of train_steps instead of number of epochs')  # noqa: E501
        parser.add_argument('--save_model_steps', dest="save_model_steps", type=int, default=500,
                            help='number of steps at which the model is periodically saved.')
        parser.add_argument('--timeline', dest="timeline", default=None, help='Trace filename for timeline')

        self.args = parser.parse_args(self.custom_args, namespace=self.args)

        # Calculate num cores and num intra threads, if the values weren't provided.
        # For a single instance run, use the number of logical cores per socket
        # for multi instance, use the number of logical cores per socket - 2
        # Note that most models use the number of physical cores for these values,
        # but this model performs better with using logical cores.
        if not self.args.num_cores or not self.args.num_intra_threads:
            num_logical_cores_per_socket = \
                platform_util.num_cores_per_socket * platform_util.num_threads_per_core

            cores_to_use = num_logical_cores_per_socket \
                if not os.environ["MPI_NUM_PROCESSES"] or int(os.environ["MPI_NUM_PROCESSES"]) <= 1 else \
                num_logical_cores_per_socket - 2

            if not self.args.num_cores or self.args.num_cores == -1:
                self.args.num_cores = cores_to_use

            if not self.args.num_intra_threads or self.args.self.args.num_intra_threads == -1:
                self.args.num_intra_threads = cores_to_use

        if not self.args.num_inter_threads:
            self.args.num_inter_threads = 1

        omp_num_threads = platform_util.num_cores_per_socket

        set_env_var("OMP_NUM_THREADS", omp_num_threads if self.args.num_cores == -1 else self.args.num_cores)

        cmd_args = " --data_dir {0}".format(self.args.data_location)
        cmd_args += " --batch_size {0}".format(self.args.batch_size)
        cmd_args += " --num_inter_threads {0}".format(self.args.num_inter_threads)
        cmd_args += " --num_intra_threads {0}".format(self.args.num_intra_threads)
        cmd_args += " --model=ssd300 --data_name coco"
        cmd_args += " --mkl=True --device=cpu --data_format=NHWC"
        cmd_args += " --variable_update=horovod --horovod_device=cpu"
        cmd_args += " --batch_group_size={0}".format(self.args.num_train_steps + 10)

        # check if user has any kmp related environmental variables set
        # if set, then pass those parameter to the training script
        if os.environ["KMP_AFFINITY"]:
            cmd_args += " --kmp_affinity={0}".format(os.environ["KMP_AFFINITY"])
        if os.environ["KMP_SETTINGS"]:
            cmd_args += " --kmp_settings={0}".format(os.environ["KMP_SETTINGS"])
        if os.environ["KMP_BLOCKTIME"]:
            cmd_args += " --kmp_blocktime={0}".format(os.environ["KMP_BLOCKTIME"])
        if (self.args.timeline is not None):
            cmd_args += " --use_chrome_trace_format=True --trace_file={0}".format(self.args.timeline)

        if (self.args.accuracy_only):
            # eval run arguments
            cmd_args += " --train_dir={0}".format(self.args.checkpoint)
            cmd_args += " --eval=true"
            cmd_args += " --num_eval_epochs=1"
            cmd_args += " --print_training_accuracy=True"
        elif (self.args.backbone_model is None):
            # benchmarking run arguments
            cmd_args += " --weight_decay {0}".format(self.args.weight_decay)
            cmd_args += " --num_warmup_batches {0}".format(self.args.num_warmup_batches)
            cmd_args += " --num_batches {0}".format(self.args.num_train_steps)

            # write checkpoints to the checkpoint dir, if there is one
            if self.args.checkpoint:
                cmd_args += " --train_dir={}".format(self.args.checkpoint)
        else:
            # convergence training arguments
            checkpoints_found = False
            for f in os.listdir(self.args.checkpoint):
                if "model.ckpt" in f:
                    checkpoints_found = True
                    break
            if checkpoints_found:
                if (self.args.backbone_model is not None):
                    print("Warning: Ignoring backbone_model since checkpoint directory is not empty. "
                          "Model will try to restore checkpoints from {}.".format(self.args.checkpoint), flush=True)
            else:
                cmd_args += " --backbone_model_path={0}".format(os.path.join(self.args.backbone_model,
                                                                             'model.ckpt-28152'))

            cmd_args += " --optimizer=momentum"
            cmd_args += " --weight_decay={0}".format(self.args.weight_decay)
            cmd_args += " --momentum=0.9"

            if self.args.epochs > 0:
                cmd_args += " --num_epochs={0}".format(self.args.epochs)
            else:
                cmd_args += " --num_batches {0}".format(self.args.num_train_steps)
            cmd_args += " --num_warmup_batches={0}".format(self.args.num_warmup_batches)
            cmd_args += " --train_dir={0}".format(self.args.checkpoint)
            cmd_args += " --save_model_steps={0}".format(self.args.save_model_steps)

        self.cmd = "{} ".format(self.python_exe)

        self.training_script_dir = os.path.join('/tmp/benchmark_ssd_resnet34/scripts/tf_cnn_benchmarks')
        training_script = os.path.join(self.training_script_dir, 'tf_cnn_benchmarks.py')

        self.cmd = self.cmd + training_script + cmd_args

    def run(self):
        original_dir = os.getcwd()
        os.chdir(self.training_script_dir)
        # Run benchmarking
        start_time = time.time()
        self.run_command(self.cmd)
        print("Total execution time: {} seconds".format(time.time() - start_time))
        os.chdir(original_dir)
