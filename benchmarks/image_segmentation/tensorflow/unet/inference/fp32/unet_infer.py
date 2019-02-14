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

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time

from argparse import ArgumentParser
from tf_unet import image_gen
from tf_unet import unet


arg_parser = ArgumentParser(
    description="The launchpad for all performance scripts.")
arg_parser.add_argument("-bs", "--batch_size", help="The batch size", type=int,
                        dest="batch_size", default=1)
arg_parser.add_argument("-p", help="do tfprofile", dest="do_tfprofile",
                        default=False)
arg_parser.add_argument("-i", "--profile_absdir",
                        help="The profile abs directory", dest="profile_path",
                        default=os.path.dirname(os.path.realpath(__file__)))
arg_parser.add_argument("-ia", "--num_intra_threads", help="The intra size",
                        type=int, dest="intra", default=28)
arg_parser.add_argument("-ie", "--num_inter_threads", help="The inter size",
                        type=int, dest="inter", default=1)
arg_parser.add_argument("-nw", "--warmup", help="warmup number", type=int,
                        dest="warmup", default=80)
arg_parser.add_argument("-nb", "--batches", help="number of batches", type=int,
                        dest="numbatch", default=400)
arg_parser.add_argument("-cp", "--ckpt", help="checkpoint", dest="ckpt_path",
                        default="./unet_trained/model.cpkt")
arg_parser.add_argument("--verbose", help="enable verbose logging",
                        dest="verbose", action="store_true")

plt.switch_backend("agg")

nx = 572
ny = 572


generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20)
verbose = arg_parser.parse_args().verbose
batchsize = arg_parser.parse_args().batch_size
intra = arg_parser.parse_args().intra
inter = arg_parser.parse_args().inter
warmup = arg_parser.parse_args().warmup
numbatch = arg_parser.parse_args().numbatch

x_test, y_test = generator(batchsize)
net = unet.Unet(channels=generator.channels, n_class=generator.n_class,
                keep_prob=1.0, layers=3, features_root=16)

if arg_parser.parse_args().do_tfprofile:
    with tf.contrib.tfprof.ProfileContext(
        arg_parser.parse_args().profile_path, trace_steps=range(200, 250, 1),
            dump_steps=[250]) as pctx:
        prediction = net.predict(arg_parser.parse_args().ckpt_path, x_test)
else:
    if warmup > 0:
        prediction = net.predict(arg_parser.parse_args().ckpt_path, x_test)
    if numbatch > 0:
        start_time = time.time()
        prediction = net.predict(arg_parser.parse_args().ckpt_path, x_test)
        infer_time = time.time() - start_time

        print("Batch size: {}\nBatches number: {}".format(batchsize, numbatch))
        print("Time spent per BATCH: %.4f ms" % (infer_time * 1000 / numbatch))
        print("Total samples/sec: %.4f samples/s" %
              (numbatch * batchsize / infer_time))
