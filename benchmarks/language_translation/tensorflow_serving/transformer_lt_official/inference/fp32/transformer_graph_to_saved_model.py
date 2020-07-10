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

"""Import a Transformer-LT model graph and export a SavedModel.

Usage: transformer_graph_to_saved_model.py [--model_version=y] import_path export_dir
"""

from __future__ import print_function

import sys
import tensorflow as tf

tf.compat.v1.app.flags.DEFINE_integer('model_version', 1, 'Version number of the model.')
tf.compat.v1.app.flags.DEFINE_string('import_path', '', 'Model import path.')
tf.compat.v1.app.flags.DEFINE_string('export_dir', '/tmp', 'Export directory.')
FLAGS = tf.compat.v1.app.flags.FLAGS


def main(_):
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
        print('Usage: transformer_graph_to_saved_model.py [--model_version=y] import_path export_dir')
        sys.exit(-1)
    if FLAGS.import_path == '':
        print('Please specify the path to the model graph you want to convert to SavedModel format.')
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print('Please specify a positive value for version number.')
        sys.exit(-1)

    # Import model graph
    with tf.compat.v1.Session() as sess:
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(FLAGS.import_path, 'rb') as input_file:
            input_graph_content = input_file.read()
            graph_def.ParseFromString(input_graph_content)

        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        sess.run(tf.compat.v1.global_variables_initializer())

        # Build the signature_def_map.
        in_data = sess.graph.get_tensor_by_name('input_tensor:0')
        inputs = {'input': tf.compat.v1.saved_model.build_tensor_info(in_data)}

        out_data = sess.graph.get_tensor_by_name('model/Transformer/strided_slice_19:0')
        outputs = {'output': tf.compat.v1.saved_model.build_tensor_info(out_data)}

        signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        # Save out the SavedModel
        print('Exporting trained model to', FLAGS.export_dir + '/' + str(FLAGS.model_version))
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(FLAGS.export_dir + '/' + str(FLAGS.model_version))
        builder.add_meta_graph_and_variables(
            sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }
        )
        builder.save()

    print('Done!')


if __name__ == '__main__':
    tf.compat.v1.app.run()
