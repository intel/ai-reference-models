# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Utility functions specifically for NMT."""
from __future__ import print_function

import codecs
import time
import numpy as np
import tensorflow as tf
import logging as log
import evaluation_utils
import misc_utils as utils


__all__ = ["decode_and_evaluate", "get_translation"]


def decode_and_evaluate(mode, sess, out_tensor, trans_file, ref_file,
                        metric='bleu', beam_width=10,
                        num_translations_per_input=1, iterations=1):
  """Decode a test set and compute a score according to the evaluation task."""
  utils.print_out("  Decoding to output %s" % trans_file)

  with codecs.getwriter("utf-8")(
      tf.io.gfile.GFile(trans_file, mode="wb")) as trans_f:
    trans_f.write("")  # Write empty string to ensure file is created.
    num_translations_per_input = min(num_translations_per_input, beam_width)

    print("  Running inference with beam_width %g, num translations per input %d. " \
          % (beam_width, num_translations_per_input))
    print("  Total iterations count %d." % iterations)

    # Warmup for the first batch to take out the very first runtime
    # session overhead.
    nmt_outputs = sess.run(out_tensor) # time x batch_size x beam_width
    nmt_outputs = nmt_outputs.transpose() # beam_width x batch_size x time
    batch_size = nmt_outputs.shape[1]
    for sent_id in range(batch_size):
      translation = get_translation(nmt_outputs[0], sent_id,
                                    tgt_eos='</s>')
      if mode == 'accuracy':
        trans_f.write((translation + b"\n").decode("utf-8"))

    # prediction time is the time for the model prediction only
    # overall time is the time for data pre-processing and data post-processing
    prediction_times = list()
    overall_start = time.time()
    num_sentences = 0
    n = 0
    while n < iterations:
      n += 1
      while True:
        try:
          start = time.time()
          nmt_outputs = sess.run(out_tensor) # time x batch_size x beam_width
          nmt_outputs = nmt_outputs.transpose() # beam_width x batch_size x time
          prediction_times.append(time.time() - start)
          batch_size = nmt_outputs.shape[1]
          num_sentences += batch_size
          for sent_id in range(batch_size):
            for beam_id in range(num_translations_per_input):
              translation = get_translation(nmt_outputs[beam_id], sent_id,
                                            tgt_eos='</s>')
              if mode == 'accuracy':
                trans_f.write((translation + b"\n").decode("utf-8"))

        except tf.errors.OutOfRangeError:
          utils.print_time(
              "  Done, num sentences %d, num translations per input %d" %
              (num_sentences, num_translations_per_input), overall_start)
          break

  overall_time = (time.time() - overall_start)
  print("\nAverage Prediction Latency: {:.5f} sec per batch.".format(
    sum(prediction_times)/float(len(prediction_times))))
  print("Overall Latency: {:.5f} sec for the entire test "
        "dataset.".format(overall_time/float(iterations)))
  print("Overall Throughput : {:.3f} sentences per sec.".format(
        num_sentences/float(overall_time)))

  # Evaluation
  if mode == 'accuracy':
    if ref_file and tf.io.gfile.exists(trans_file):
      score = evaluation_utils.evaluate(ref_file, trans_file, metric)
      utils.print_out("  Accuracy metric %s: %.1f" % (metric, score))


def get_translation(nmt_outputs, sent_id, tgt_eos):
  """Given batch decoding outputs, select a sentence and turn to text."""
  if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
  # Select a sentence
  output = nmt_outputs[sent_id, :].tolist()

  # If there is an eos symbol in outputs, cut them at that point.
  if tgt_eos and tgt_eos in output:
    output = output[:output.index(tgt_eos)]
    translation = utils.format_text(output)

  return translation
