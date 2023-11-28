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
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

import sys
import time
import argparse

AUTOTUNE = tf.data.AUTOTUNE

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

        arg_parser.add_argument("--use_real_data",
                                help='Specify if to use DIV2K dataset for benchmarking.',
                                dest="use_real_data", default=False)

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

        self.load_graph()
        self.input_tensor = self.infer_graph.get_tensor_by_name(self.args.input_layer + ":0")
        self.output_tensors = self.infer_graph.get_tensor_by_name(self.args.output_layer + ":0")

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
                output_graph = optimize_for_inference(graph_def, [self.args.input_layer],
                                    [self.args.output_layer], dtypes.float32.as_datatype_enum, False)
            tf.import_graph_def(output_graph, name='')

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
        ]  # 48x48
        highres_img_cropped = highres_img[
            highres_height : highres_height + hr_crop_size,
            highres_width : highres_width + hr_crop_size,
        ]  # 144x144

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

    def benchmark(self):
        with tf.compat.v1.Session(graph=self.infer_graph, config=self.config) as sess:
            if self.args.use_real_data:
                print("Running Benchmark on Image Data!")
                self.build_data_sess()
                low_res, high_res = self.data_sess.run([self.low, self.high])
                image_data = low_res
            else:
                print("Running Benchmark on Dummy Data!")
                input_shape = [self.args.batch_size, 48, 48, 3]
                images = tf.random.truncated_normal(
                    input_shape,
                    dtype=tf.float32,
                    stddev=10,
                    name='synthetic_images')
                image_data = sess.run(images)
            
            sys.stdout.flush()
            print("[Running warmup steps...]")
            for t in range(self.args.warmup_steps):
                start_time = time.perf_counter()
                sess.run(self.output_tensors, {self.input_tensor: image_data})
                elapsed_time = time.perf_counter() - start_time
                if((t+1) % 10 == 0):
                    print("steps = {0}, {1} images/sec"
                        "".format(t+1, self.args.batch_size/elapsed_time))

            print("[Running benchmark steps...]")
            total_time   = 0
            total_images = 0
            for t in range(self.args.steps):
                start_time = time.perf_counter()
                results = sess.run(self.output_tensors, {self.input_tensor: image_data})
                elapsed_time = time.perf_counter() - start_time
                total_time = total_time + elapsed_time

                if((t+1) % 10 == 0):
                    print("steps = {0}, {1} images/sec"
                        "".format(t+1, self.args.batch_size/elapsed_time));

            print('Average Throughput: %0.2f images/s on %d iterations'%(
                self.args.batch_size * self.args.steps / total_time, self.args.steps))

    def run(self):
        self.benchmark()

if __name__ == "__main__":
  infer = model_infer()
  infer.run()

