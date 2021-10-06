#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys

import tensorflow as tf
from model import model_params
from model import transformer

tf.compat.v1.disable_v2_behavior()

_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 4
_ALPHA = 0.6

def main(unused_argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  if FLAGS.params == "base":
    params = model_params.TransformerBaseParams
  elif FLAGS.params == "big":
    params = model_params.TransformerBigParams
  else:
    raise ValueError("Invalid parameter set defined: %s."
                     "Expected 'base' or 'big.'" % FLAGS.params)
  # Set up estimator and params
  params.beam_size = _BEAM_SIZE
  params.alpha = _ALPHA
  params.extra_decode_length = _EXTRA_DECODE_LENGTH
  params.frozen_graph = None
  input_shape = [None, None]
  input_tokens = tf.compat.v1.placeholder(tf.int64, input_shape,
                                          name='input_tokens')
  with tf.compat.v1.variable_scope("model"):
    model = transformer.Transformer(params, False) 
    output = model(input_tokens)

  # Restore variables from checkpoint
  sess = tf.compat.v1.Session()
  latest_model = tf.train.latest_checkpoint(FLAGS.model_dir)
  saver = tf.compat.v1.train.Saver()
  saver.restore(sess, latest_model)

  # Freeze the graph
  graph_def = sess.graph.as_graph_def()
  output_names = ['model/Transformer/strided_slice_15',
                  'model/Transformer/strided_slice_16']
  graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
        graph_def, output_names)
  print("pb_path is", FLAGS.pb_path)
  with tf.compat.v1.gfile.GFile(FLAGS.pb_path, 'wb') as pb_file:
    pb_file.write(graph_def.SerializeToString())


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # Model arguments
  parser.add_argument(
     "--model_dir", "-md", type=str, required=True,
      help="Directory containing Transformer model checkpoints.",
      metavar="<MD>")
  parser.add_argument(
      "--pb_path", "-pp", type=str, required=True,
      help="The output pb file path",
      metavar="<PP>")
  parser.add_argument(
      "--params", "-p", type=str, default="big", choices=["base", "big"],
      help="[default: %(default)s] Parameter used for trained model.",
      metavar="<P>")

  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv)
