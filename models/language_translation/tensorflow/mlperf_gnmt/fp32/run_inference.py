import codecs
import argparse
import os
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

import misc_utils as utils
from nmt_utils import decode_and_evaluate

from tensorflow_addons import seq2seq

parser = argparse.ArgumentParser()
parser.add_argument("--in_graph", type=str, required=True,
                    help="Specify the frozen inference graph in pb format.")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Specify inference batch size.")
parser.add_argument("--num_inter_threads", type=int, default=0,
                   help="Specify number of inter-op threads.")
parser.add_argument("--num_intra_threads", type=int, default=0,
                   help="Specify number of intra-op threads.")
parser.add_argument("--src_vocab_file", type=str, required=True,
                    help="Specify source vocab file.")
parser.add_argument("--tgt_vocab_file", type=str, required=True,
                    help="Specify target vocabulary file.")
parser.add_argument("--inference_input_file", type=str, required=True,
                    help="Specify input file to be translated.")
parser.add_argument("--inference_output_file", type=str, default=None,
                    help="Specify output file for resulting translation.")
parser.add_argument("--inference_ref_file", type=str, required=True,
                    help="Specify reference output file.")
parser.add_argument("--run", type=str, default="accuracy",
                    help="Specify either 'accuracy' for BLEU metric or "
                         "'performance' for latency and throughput.")
args = parser.parse_args()

out_dir = os.path.join(os.getcwd(), 'output')
tf.io.gfile.makedirs(out_dir)

if args.inference_output_file:
  inference_output_file = args.inference_output_file
else:
  inference_output_file = os.path.join(out_dir, 'gnmt-out')

def read_source_sentences(inference_input_file):
  """Load inference data."""
  with codecs.getreader("utf-8")(
      tf.io.gfile.GFile(inference_input_file, mode="rb")) as f:
    inference_data = f.read().splitlines()
  return inference_data

def create_new_vocab_file(vocab_file):
  """Creates a new vocabulary file prepending three new tokens:
  (1) <unk> for unknown tag, (2) <s> for start of sentence tag, and (3) </s> for end of
  sentence tag."""
  vocab = []
  with codecs.getreader("utf-8")(tf.io.gfile.GFile(vocab_file, "rb")) as f:
    vocab_size = 0
    for word in f:
      vocab_size += 1
      vocab.append(word.strip())

  if tf.io.gfile.exists(vocab_file):
    utils.print_out("# Vocab file %s exists" % vocab_file)
    assert len(vocab) >= 3
    (unk, sos, eos) = ("<unk>", "<s>", "</s>")
    if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
      utils.print_out("The first 3 vocab words [%s, %s, %s]"
                      " are not [%s, %s, %s]" %
                      (vocab[0], vocab[1], vocab[2], unk, sos, eos))
      vocab = [unk, sos, eos] + vocab
      vocab_size += 3
      new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
      with codecs.getwriter("utf-8")(
          tf.io.gfile.GFile(new_vocab_file, "wb")) as f:
        for word in vocab:
          f.write("%s\n" % word)
      vocab_file = new_vocab_file
  else:
    raise ValueError("vocab_file '%s' does not exist." % vocab_file)
  return vocab_file

if __name__ == "__main__":
  graph_def = graph_pb2.GraphDef()
  with tf.io.gfile.GFile(args.in_graph, "rb") as f:
    data = f.read()
  graph_def.ParseFromString(data)
  graph = tf.Graph()
  with graph.as_default():
    importer.import_graph_def(graph_def, input_map={}, name="")
    # Get input and output and tensors/ops for inference.
    src_vocab_placeholder = graph.get_tensor_by_name('source_vocab_file:0')
    tgt_vocab_placeholder = graph.get_tensor_by_name('target_vocab_file:0')
    src_data_placeholder = graph.get_tensor_by_name('source_data:0')
    batch_size_placeholder = graph.get_tensor_by_name('batch_size:0')

    tables_initializer = graph.get_operation_by_name('init_all_tables')
    iterator_initilizer = graph.get_operation_by_name('MakeIterator')
    sample_words_tensor = graph.get_tensor_by_name('hash_table_Lookup_1/LookupTableFindV2:0')

  # Create a session with imported graph.
  config_proto = tf.compat.v1.ConfigProto(allow_soft_placement=True,
      intra_op_parallelism_threads = args.num_intra_threads,
      inter_op_parallelism_threads = args.num_inter_threads)
  sess = tf.compat.v1.Session(graph=graph, config=config_proto)

  # Read source data.
  src_data = read_source_sentences(args.inference_input_file)

  # Initialize vocabulary tables and source data iterator.
  sess.run(tables_initializer, feed_dict={
      src_vocab_placeholder: create_new_vocab_file(args.src_vocab_file),
      tgt_vocab_placeholder: create_new_vocab_file(args.tgt_vocab_file)})
  sess.run(iterator_initilizer, feed_dict={
      src_data_placeholder: src_data,
      batch_size_placeholder: args.batch_size})

  # Decode
  decode_and_evaluate(args.run, sess, sample_words_tensor, inference_output_file,
                      args.inference_ref_file)
