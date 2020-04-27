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
#

import tensorflow as tf
from tensorflow.python.framework import ops
from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
import numpy as np
from utils import tokenizer
from utils.tokenizer import Subtokenizer

from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import app
import time
import pandas as pd
from timeit import default_timer as timer

flags = flags_lib
FLAGS = flags.FLAGS

flags.DEFINE_string("in_graph", 'fp32_graphdef.pb',
                    """TensorFlow 'GraphDef' file for FP32 to load.""")
flags.DEFINE_bool("input_binary", True,
                  """Whether the input files are in binary format.""")
flags.DEFINE_string(
      "vocab_file", "vocab.ende.32768",
              "Path to subtoken vocabulary file.")
flags.DEFINE_string(
      "file", "newstest2014.en",
              """File saved to an output file.""")
flags.DEFINE_string(
    "file_out", "translate.txt",
          """If --file flag is specified, save translation to this file.""")
flags.DEFINE_integer("batch_size", 64,
                     """The validation batch size""")
flags.DEFINE_integer("num_inter", 1,
                     """Number of sentences to exclude from validation file.""")
flags.DEFINE_integer("num_intra", 40,
                     """Number of sentences to exclude from validation file.""")
flags.DEFINE_integer("num_batches", 0,
                     """Number of batches of sentences to run inference for""")
flags.DEFINE_bool("sort_input_sentences", None,
                  """Sort the sequence of sentences in validation file. 
                  Sorting improves batch processing time""")

def input_generator_ts():
  """Read and sort lines based on token count from the file
     sorted by decreasing length based on token sorting.
  Args:
    filename: String name of file to read inputs from.
  Returns:
    Sorted list of inputs, and dictionary mapping original index->sorted index
    of each element.
  """
  with tf.io.gfile.GFile(FLAGS.file) as f:
    records = f.read().split("\n")
    inputs = [record.strip() for record in records]
    if not inputs[-1]:
      inputs.pop()

  subtokenizer = Subtokenizer(FLAGS.vocab_file)

  batch = []
  token_lens=[]
  for i, line in enumerate(inputs):
    enc = subtokenizer.encode(line, add_eos=True)
    token_lens.append((i, len(enc)))

  sorted_by_token_input_lens = sorted(token_lens, key=lambda x: x[1], reverse=True)

  #print('sorted_by_token_input_lens:{}'.format(sorted_by_token_input_lens))

  sorted_inputs = [None] * len(sorted_by_token_input_lens)
  sorted_keys = [0] * len(sorted_by_token_input_lens)

  for i, (index, _) in enumerate(sorted_by_token_input_lens):
    sorted_inputs[i] = inputs[index]
    sorted_keys[index] = i
    enc=subtokenizer.encode(sorted_inputs[i], add_eos=True)
    batch.append(enc)

  return batch,sorted_keys

def input_generator_ws():
  """Read and sort lines from the file sorted by decreasing length based on word counts.
  Args:
    filename: String name of file to read inputs from.
  Returns:
    Sorted list of inputs, and dictionary mapping original index->sorted index
    of each element.
  """
  with tf.io.gfile.GFile(FLAGS.file) as f:
    records = f.read().split("\n")
    inputs = [record.strip() for record in records]
    if not inputs[-1]:
      inputs.pop()

  batch = []

  subtokenizer = Subtokenizer(FLAGS.vocab_file)

  input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=lambda x: x[1], reverse=True)

  sorted_inputs = [None] * len(sorted_input_lens)
  sorted_keys = [0] * len(sorted_input_lens)
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs[i] = inputs[index]
    sorted_keys[index] = i
    enc=subtokenizer.encode(sorted_inputs[i], add_eos=True)
    batch.append(enc)
  return batch,sorted_keys

def _trim_and_decode(ids):
  """Trim EOS and PAD tokens from ids, and decode to return a string."""
  subtokenizer = Subtokenizer(FLAGS.vocab_file)
  try:
    index = list(ids).index(tokenizer.EOS_ID)
    return subtokenizer.decode(ids[:index])
  except ValueError:  # No EOS found in sequence
    return subtokenizer.decode(ids)

def main(unused_args):

  graph_def = graph_pb2.GraphDef()
  graph_file=FLAGS.in_graph


  start=timer()
  with open(graph_file, "rb") as f:
    if FLAGS.input_binary:
      graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), graph_def)
  end=timer()
  graph_parse_time = end-start
  print("Graph parsed in %f s" % (end-start))

  start=timer()
  with tf.Graph().as_default() as graph:
    y = tf.import_graph_def(graph_def,return_elements=["model/Transformer/strided_slice_19:0"], name='')
  end=timer()
  print("import_graph_def took %fs" % (end-start))

  start=timer()
  batches,sorted_keys = input_generator_ts()
  end=timer()
  sort_time = end-start
  print("tokenizer took %f s" % (sort_time))

  DATASET_SIZE=len(batches)
  print("Translating {} sentences from English to German.".format(DATASET_SIZE))

  session_config = tf.compat.v1.ConfigProto(
      inter_op_parallelism_threads=FLAGS.num_inter,
      intra_op_parallelism_threads=FLAGS.num_intra)

  with tf.compat.v1.Session(config=session_config, graph=graph) as sess:

    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    translations = []

    batch = []
    batch_num=0
    line_count=0

    total_start_time = time.time()
    inference_time = 0.0

    for i, line in enumerate(batches): 
      batch.append(line)
      duration = 0.0
      if i % FLAGS.batch_size == 0:
        line_count += 1
        batch_num = (i // FLAGS.batch_size) + 1
        start_time = time.time()
        dec_tensor = sess.run(y, feed_dict={'input_tensor:0': pd.DataFrame(batch).fillna(0).values.astype(np.int32)})
        duration = time.time() - start_time
        translations.append(dec_tensor)
        #print('Batch inferencing time:%s for batch size:%d and batch:%d' % (duration, FLAGS.batch_size, batch_num))
        batch = []
      elif i % (len(batches) - 1) == 0:
        batch_num = (i // FLAGS.batch_size) + 1
        start_time = time.time()
        dec_tensor = sess.run(y, feed_dict={'input_tensor:0': pd.DataFrame(batch).fillna(0).values.astype(np.int32)})
        duration = time.time() - start_time
        translations.append(dec_tensor)
        #print('Batch inferencing time:%s for batch size:%d and batch:%d' % (duration, FLAGS.batch_size, batch_num))
        batch = []
      inference_time += duration

    inference_time += graph_parse_time
    inference_time += sort_time
    print('Total inferencing time:%s' %(inference_time))
    print('Throughput:{} sentences/second'.format((DATASET_SIZE)/inference_time))

    translation_count = 0

    decoded_translations=[]
    for i,tr in enumerate(translations):
      for j,itr in enumerate(tr):
        for k,otr in enumerate(itr):
          translation_count += 1
          decoded_translations.append(_trim_and_decode(otr))

    print('Total number of sentences translated:%d' % (translation_count))

    with tf.io.gfile.GFile(FLAGS.file_out, "w") as f:
      for i in sorted_keys:
        f.write("%s\n" % decoded_translations[i])

if __name__ == "__main__":
  app.run()
