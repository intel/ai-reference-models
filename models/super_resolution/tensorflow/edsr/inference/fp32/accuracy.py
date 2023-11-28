#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import time

from tensorflow.core.protobuf import rewriter_config_pb2

import argparse

EDSR_NUM_VAL_IMAGES = 100
class model_infer:

    def __init__(self):
        arg_parser = argparse.ArgumentParser(description='Parse args')

        arg_parser.add_argument("--batch_size",
                                help="Specify the batch size. If this " \
                                    "parameter is not specified or is -1, the " \
                                    "largest ideal batch size for the model will " \
                                    "be used.",
                                dest="batch_size", type=int, default=-1)

        arg_parser.add_argument("--num_inter_threads",
                                help='The number of inter-thread.',
                                dest='num_inter_threads', type=int, default=0)

        arg_parser.add_argument("--num_intra_threads",
                                help='The number of intra-thread.',
                                dest='num_intra_threads', type=int, default=0)

        arg_parser.add_argument("--input_graph",
                                help='Specify the input graph.',
                                dest='input_graph')

        arg_parser.add_argument("--accuracy-only",
                                help='For accuracy measurement only.',
                                dest='accuracy_only', action='store_true')

        arg_parser.add_argument("--steps",
                                help='For accuracy measurement only.',
                                dest='steps', default=50, type=int)

        arg_parser.add_argument("--warmup_steps",
                                help='For accuracy measurement only.',
                                dest='warmup_steps', default=10, type=int)

        arg_parser.add_argument("--precision",
                                help='Specify the model precision to use: fp32',
                                dest='precision', default='fp32', choices=['fp32'])
        
        arg_parser.add_argument("--input_layer",
                                help="name of input layer",
                                dest="input_layer", default="IteratorGetNext")
        
        arg_parser.add_argument("--output_layer",
                                help="name of output layer",
                                dest="output_layer", default="NHWC_output")

        # parse the arguments
        self.args = arg_parser.parse_args()

        self.config = tf.compat.v1.ConfigProto()
        self.config.intra_op_parallelism_threads = self.args.num_intra_threads
        self.config.inter_op_parallelism_threads = self.args.num_inter_threads
        self.config.use_per_session_threads = 1
        self.config.graph_options.rewrite_options.CopyFrom(
            rewriter_config_pb2.RewriterConfig(
                remapping=rewriter_config_pb2.RewriterConfig.ON,
                constant_folding=rewriter_config_pb2.RewriterConfig.AGGRESSIVE))
        
        self.load_graph()
        self.input_tensor = self.infer_graph.get_tensor_by_name(self.args.input_layer + ":0")
        self.output_tensors = self.infer_graph.get_tensor_by_name(self.args.output_layer + ":0")
        self.scale = 3

    def build_data_sess(self):
        data_graph = tf.Graph()
        with data_graph.as_default():
            self.low, self.high = self.get_input()
            self.data_sess = tf.compat.v1.Session(graph=data_graph, config=self.config)

    def load_graph(self):
        print('load graph from: ' + str(self.args.input_graph))

        self.infer_graph = tf.Graph()
        with self.infer_graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.FastGFile(self.args.input_graph, 'rb') as input_file:
                input_graph_content = input_file.read()
                graph_def.ParseFromString(input_graph_content)
            tf.import_graph_def(graph_def, name='')
    

    def random_crop(self, lowres_img, highres_img, hr_crop_size=144, scale=3):
        """Crop images.

        low resolution images: 48x48
        high resolution images: 144x144
        """
        lowres_crop_size = hr_crop_size // scale  # 144//3=48
        lowres_img_shape = tf.shape(lowres_img)[:2]  # (height,width)

        lowres_width = tf.random.uniform(
            shape=(), maxval=lowres_img_shape[1] - lowres_crop_size + 1, dtype=tf.int32
        )
        lowres_height = tf.random.uniform(
            shape=(), maxval=lowres_img_shape[0] - lowres_crop_size + 1, dtype=tf.int32
        )

        highres_width = lowres_width * scale
        highres_height = lowres_height * scale

        lowres_img_cropped = lowres_img[
            lowres_height : lowres_height + lowres_crop_size,
            lowres_width : lowres_width + lowres_crop_size,
        ]  # 32x32
        highres_img_cropped = highres_img[
            highres_height : highres_height + hr_crop_size,
            highres_width : highres_width + hr_crop_size,
        ]  # 96x96

        return lowres_img_cropped, highres_img_cropped

    def dataset_object(self, dataset_cache):

        ds = dataset_cache
        ds = ds.map(
            lambda lowres, highres: self.random_crop(lowres, highres, scale=3),
            num_parallel_calls=AUTOTUNE,
        )

        # Batching Data
        ds = ds.batch(self.args.batch_size)
        # prefetching allows later images to be prepared while the current image is being processed
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def get_input(self):
        div2k_data = tfds.image.Div2k(config="bicubic_x3")
        div2k_data.download_and_prepare()
        val = div2k_data.as_dataset(split="validation", as_supervised=True, shuffle_files=False)
        val_cache = val.cache()
        val_ds = self.dataset_object(val_cache)
        ds_iterator = tf.compat.v1.data.make_one_shot_iterator(val_ds)
        low, high = ds_iterator.get_next()
        return low, high
    
    def calcmeanImages(self, images, bgr=False):
        """
        Calculates the mean of a dataset.
        """
        total_mean = [0, 0, 0]
        im_counter = 0
        for image in images:
            mean_rgb = np.mean(image, axis=(0, 1), dtype=np.float64)
            total_mean += mean_rgb
            im_counter += 1
        total_mean[:] = [x / im_counter for x in total_mean]

        # rgb to bgr
        if bgr is True:
            total_mean = total_mean[...,::-1]

        return total_mean
    
    def accuracy_check(self):
        print("Inference for accuracy check.")
        self.build_data_sess()
        with tf.compat.v1.Session(graph=self.infer_graph, config=self.config) as sess:
            iter = 0
            val_psnr = np.array([])
            val_ssim = np.array([])
            total_time = 0
            while True:
                print('Run {0} iter'.format(iter))
                iter += 1
                low_res, high_res = self.data_sess.run([self.low, self.high])
                floatimg = low_res.astype(np.float32)
                batch_mean = self.calcmeanImages(floatimg)
                floatimg = floatimg - batch_mean
                start_time = time.perf_counter()
                output = sess.run(self.output_tensors, feed_dict={self.input_tensor: floatimg})
                elapsed_time = time.perf_counter() - start_time
                total_time += elapsed_time
                output = (output + batch_mean).clip(min=0, max=255)
                output = (output).astype(np.uint8)
                val_psnr = np.concatenate((val_psnr, tf.image.psnr(output, high_res, max_val=255).eval()), axis=None)
                val_ssim = np.concatenate((val_ssim, tf.image.ssim(output, high_res, max_val=255, filter_size=11,
                           filter_sigma=1.5, k1=0.01, k2=0.03).eval()), axis=None)
                print("Iteration time: %0.4f ms" % elapsed_time)
                if iter * self.args.batch_size >= EDSR_NUM_VAL_IMAGES:
                    print("Average PSNR: "+ str(np.mean(val_psnr)))
                    print("Average SSIM: " + str(np.mean(val_ssim)))
                    print("Average Time per Iteration: %0.4f ms" % (total_time/iter))
                    break

    def run(self):
        self.accuracy_check()

if __name__ == "__main__":
  infer = model_infer()
  infer.run()

