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

"""Import Mask R-CNN model weights and export a SavedModel.

Usage: mask_rcnn_weights_to_saved_model.py [--model_version=y] import_path export_dir
"""

from __future__ import print_function

import os
import sys
import tensorflow as tf
import keras.backend as K

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from config import Config
import model as modellib


tf.app.flags.DEFINE_integer('model_version', 1, 'Version number of the model.')
tf.app.flags.DEFINE_string('import_path', '', 'Model import path.')
tf.app.flags.DEFINE_string('export_dir', '/tmp', 'Export directory.')
FLAGS = tf.app.flags.FLAGS


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 #int(args.infbs)
    DETECTION_MIN_CONFIDENCE = 0

def freeze_session(session, output_names=None):
    graph = session.graph
    with graph.as_default():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        for node in input_graph_def.node:
            node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def main(_):
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
        print('Usage: mask_rcnn_weights_to_saved_model.py [--model_version=y] import_path export_dir')
        sys.exit(-1)
    if FLAGS.import_path == '':
        print('Please specify the path to the model weights you want to convert to SavedModel format.')
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print('Please specify a positive value for version number.')
        sys.exit(-1)

    config = InferenceConfig()
    config.NUM_INTRA = 16
    config.NUM_INTER = 1

    # Construct model and import weights
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=FLAGS.import_path)
    model.load_weights(FLAGS.import_path, by_name=True)

    # Freeze session
    output_names = [output.op.name for output in model.keras_model.outputs]
    frozen_graph = freeze_session(K.get_session(), output_names=output_names)

    export_dir = os.path.join(FLAGS.export_dir, str(FLAGS.model_version))
    signature = {}

    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(frozen_graph, name='')
        graph_def = tf.get_default_graph()
        input_image = graph_def.get_tensor_by_name('input_image:0')
        input_image_meta = graph_def.get_tensor_by_name('input_image_meta:0')

        output_detection = graph_def.get_tensor_by_name('mrcnn_detection/Reshape_1:0')
        output_mask = graph_def.get_tensor_by_name('mrcnn_mask/Reshape_1:0')

        signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {'input_image': input_image, 'input_image_meta': input_image_meta},
                {'detection': output_detection, 'mask': output_mask})

        # Save out the SavedModel
        print('Exporting trained model to {}'.format(export_dir))
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=signature)
        builder.save()

    print('Done!')

if __name__ == '__main__':
    tf.app.run()
