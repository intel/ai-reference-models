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

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Evaluates a TFGAN trained CIFAR model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import data_provider
import networks
import util
import time
import os
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import monitored_session

flags = tf.flags
FLAGS = tf.flags.FLAGS
tfgan = tf.contrib.gan

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('ckpt', './cifar-model/unconditional',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', './cifar-model/eval',
                    'Directory where the results are saved to.')

# flags.DEFINE_string('dataset_dir', None, 'Location of data.')
flags.DEFINE_string('dl', None, 'Location of data.')

flags.DEFINE_integer('num_inception_images', 10,
                     'The number of images to run through Inception at once.')

flags.DEFINE_boolean('eval_real_images', True,
                     'If `True`, run Inception network on real images.')

flags.DEFINE_boolean('conditional_eval', False,
                     'If `True`, set up a conditional GAN.')

flags.DEFINE_boolean('eval_frechet_inception_distance', True,
                     'If `True`, compute Frechet Inception distance using real '
                     'images and generated images.')

flags.DEFINE_integer('num_images_per_class', 10,
                     'When a conditional generator is used, this is the number '
                     'of images to display per class.')

flags.DEFINE_boolean('write_to_disk', True, 'If `True`, run images to disk.')

# parameter
flags.DEFINE_integer('nw', 20,
                     'number of warm up')
flags.DEFINE_integer('nb', 1000,
                     'number of batches to run')
# opmization parameters
flags.DEFINE_integer('num_intra_threads', 28,
                     'Specifiy the number threads within layers')
flags.DEFINE_integer('num_inter_threads', 1,
                     'Specify the number threads between layers')
flags.DEFINE_integer('kmp_settings', 0,
                     'If set to 1, MKL settings will be printed.')
flags.DEFINE_integer('kmp_blocktime', 1,
                     'The time, in milliseconds, that a thread should wait, after completing the execution of a parallel region, before sleepig.')
flags.DEFINE_integer('bs', 100,
                     'the batchsize')


# flags.DEFINE_integer('dl', None,
#                     'the data location')

def mkl_setup():
    if not os.environ.get('KMP_BLOCKTIME'):
        os.environ['KMP_BLOCKTIME'] = str(FLAGS.kmp_blocktime)
    if not os.environ.get('KMP_SETTINGS'):
        os.environ['KMP_SETTINGS'] = str(FLAGS.kmp_settings)
    if not os.environ.get('KMP_AFFINITY'):
        os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
    if FLAGS.num_intra_threads > 0 and not os.environ.get('OMP_NUM_THREADS'):
        os.environ['OMP_NUM_THREADS'] = str(FLAGS.num_intra_threads)


def create_config_proto():
    """Returns session config proto.
    Args:
      params: Params tuple, typically created by make_params or
              make_params_from flags.
    """
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = FLAGS.num_intra_threads
    config.inter_op_parallelism_threads = FLAGS.num_inter_threads
    return config


def main(_, run_eval_loop=True):
    # Fetch and generate images to run through Inception.
    with tf.name_scope('inputs'):
        real_data, num_classes = _get_real_data(
            # FLAGS.bs, FLAGS.dataset_dir)
            FLAGS.bs, FLAGS.dl)
        generated_data = _get_generated_data(
            FLAGS.bs, FLAGS.conditional_eval, num_classes)

    # Compute Frechet Inception Distance.
    if FLAGS.eval_frechet_inception_distance:
        fid = util.get_frechet_inception_distance(
            real_data, generated_data, FLAGS.bs,
            FLAGS.num_inception_images)
        tf.summary.scalar('frechet_inception_distance', fid)

    # Compute normal Inception scores.
    if FLAGS.eval_real_images:
        inc_score = util.get_inception_scores(
            real_data, FLAGS.bs, FLAGS.num_inception_images)
    else:
        inc_score = util.get_inception_scores(
            generated_data, FLAGS.bs, FLAGS.num_inception_images)
    tf.summary.scalar('inception_score', inc_score)

    # If conditional, display an image grid of difference classes.
    if FLAGS.conditional_eval and not FLAGS.eval_real_images:
        reshaped_imgs = util.get_image_grid(
            generated_data, FLAGS.bs, num_classes,
            FLAGS.num_images_per_class)
        tf.summary.image('generated_data', reshaped_imgs, max_outputs=1)

    # Create ops that write images to disk.
    image_write_ops = None
    if FLAGS.conditional_eval and FLAGS.write_to_disk:
        reshaped_imgs = util.get_image_grid(
            generated_data, FLAGS.bs, num_classes,
            FLAGS.num_images_per_class)
        uint8_images = data_provider.float_image_to_uint8(reshaped_imgs)
        image_write_ops = tf.write_file(
            '%s/%s' % (FLAGS.eval_dir, 'conditional_cifar10.png'),
            tf.image.encode_png(uint8_images[0]))
    else:
        if FLAGS.bs >= 100 and FLAGS.write_to_disk:
            reshaped_imgs = tfgan.eval.image_reshaper(
                generated_data[:100], num_cols=FLAGS.num_images_per_class)
            uint8_images = data_provider.float_image_to_uint8(reshaped_imgs)
            image_write_ops = tf.write_file(
                '%s/%s' % (FLAGS.eval_dir, 'unconditional_cifar10.png'),
                tf.image.encode_png(uint8_images[0]))

    # For unit testing, use `run_eval_loop=False`.
    if not run_eval_loop: return

    checkpoint_path = tf_saver.latest_checkpoint(FLAGS.ckpt)
    if (checkpoint_path is None):
        print("error, no checkpoint path")

    # set mkl env
    mkl_setup()
    sess_config = create_config_proto()

    session_creator = monitored_session.ChiefSessionCreator(
        scaffold=None,
        checkpoint_filename_with_path=checkpoint_path,
        master=FLAGS.master,
        config=sess_config
    )
    with monitored_session.MonitoredSession(
            session_creator=session_creator, hooks=None) as session:
        eval_ops = image_write_ops
        for warmup_i in range(FLAGS.nw):
            session.run(eval_ops, feed_dict=None)
        start = time.time()
        for batch_i in range(FLAGS.nb):
            session.run(eval_ops, feed_dict=None)
        end = time.time()
        inference_time = end - start

        print("Batch size:", FLAGS.bs, "\nBatches number:", FLAGS.nb)
        print("Time spent per BATCH: %.4f ms" % (inference_time * 1000 / (FLAGS.nb)))
        print("Total samples/sec: %.4f samples/s" % (FLAGS.nb * FLAGS.bs / inference_time))


def _get_real_data(num_images_generated, dataset_dir):
    """Get real images."""
    data, _, _, num_classes = data_provider.provide_data(
        num_images_generated, dataset_dir)
    return data, num_classes


def _get_generated_data(num_images_generated, conditional_eval, num_classes):
    """Get generated images."""
    noise = tf.random_normal([num_images_generated, 64])
    # If conditional, generate class-specific images.
    if conditional_eval:
        conditioning = util.get_generator_conditioning(
            num_images_generated, num_classes)
        generator_inputs = (noise, conditioning)
        generator_fn = networks.conditional_generator
    else:
        generator_inputs = noise
        generator_fn = networks.generator
    # In order for variables to load, use the same variable scope as in the
    # train job.
    with tf.variable_scope('Generator'):
        data = generator_fn(generator_inputs, is_training=False)

    return data


if __name__ == '__main__':
    tf.app.run()