import numpy
from data_iterator import DataIterator
import tensorflow as tf
# from model import *
import time
import random
import sys
from utils import *

import argparse

from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import rewriter_config_pb2

import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', help="mode, train or test")
parser.add_argument("--seed", type=int, default=3, help="seed value")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--data_type", type=str, default='fp32', help="data type: fp32, fp16 or bfloat16")
parser.add_argument("--num_intra_threads", type=int, default=None, help="num-intra-threads")
parser.add_argument("--num_inter_threads", type=int, default=None, help="num-inter-threads")
parser.add_argument("--num_iterations", type=int, default=2, help="num_iterations")
parser.add_argument("--data_location", type=str, default=None, help="data location")
parser.add_argument("--timeline", type=bool, default=True, help="obtain timeline")
parser.add_argument("--input_graph", type=str, default=None, help="pb location")
parser.add_argument("--accuracy_only", action='store_true', help="Show accuracy only")
parser.add_argument("--exact_max_length", type=int, default=0, help="Show perf for exact max length")
parser.add_argument("--graph_type", type=str, default='static', help="graph_type: static or dynamic")

args = parser.parse_args()

def prepare_data(input, target, maxlen=None, return_neg=False, maxlen_padding=False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]
    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x)
    if maxlen_padding:
      maxlen_x = max(maxlen, maxlen_x)

    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    if args.data_type == 'fp32' or args.data_type == 'bfloat16':
        data_type = 'float32'
    elif args.data_type == 'fp16':
        data_type = 'float16'
    else:
        raise ValueError("Invalid model data type: %s" % args.data_type)
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype(data_type)
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)


def filtered_data(test_data, exact_max_length=0):
    nums = 0
    total_data = []
    prep_data = []

    print("Preparing Data....")
    mlen_padding = (exact_max_length != 0)
    prepare_start = time.time()

    for src, tgt in test_data:

        sys.stdout.flush()
        #uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(
        #    src, tgt, return_neg=True)
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(
             src, tgt, maxlen=100, return_neg=True, maxlen_padding=mlen_padding)

        #if len(mid_his[0]) < exact_max_length:
        if exact_max_length !=0 and len(mid_his[0]) < exact_max_length:
           continue

        nums += 1
        total_data.append([uids, mids, cats, mid_his, cat_his, mid_mask, target, sl])

    prepare_end = time.time()
    print("Data preperation Complete....!")
    print("Time taken for data prep :", (prepare_end - prepare_start))
    return total_data, nums

def calculate(sess, total_data, input_tensor, output_tensor, batch_size, ignore_count=0):
    #nums = 0
    #total_data = []

    #prepare_start = time.time()
    #for src, tgt in test_data:

    #    sys.stdout.flush()
    #    #uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(
    #    #    src, tgt, return_neg=True)
    #    uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(
    #         src, tgt, maxlen=100, return_neg=True)

    #    #if len(mid_his[0]) < exact_max_length:
    #   if exact_max_length !=0 and len(mid_his[0]) < exact_max_length:
    #      continue

    #   nums += 1
    #   total_data.append([uids, mids, cats, mid_his, cat_his, mid_mask, target, sl])

    #repare_end = time.time()
    ## print("prepare time   ", prepare_end - prepare_start)

    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    stored_arr = []

    eval_time = 0
    elapsed_time_records = []

    sample_freq = 9999999
    options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    total_samples = len(total_data)
    for i in range(1, total_samples + 1):

        feed_data = total_data[i - 1]
        start_time = time.time()
        if args.timeline and i % sample_freq == 0:
            prob, acc = sess.run(output_tensor,
                                 options=options,
                                 run_metadata=run_metadata,
                                 feed_dict=dict(zip(input_tensor, feed_data)))
        else:
            prob, acc = sess.run(output_tensor,
                                 feed_dict=dict(zip(input_tensor, feed_data)))
        end_time = time.time()
        if ignore_count <=0 or (ignore_count >0 and i > ignore_count):
          eval_time += end_time - start_time
          elapsed_time_records.append(end_time - start_time)
        #print("evaluation time of one batch: %.3f" % (end_time - start_time))

        target = feed_data[6]

        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()

        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])

    if args.timeline:
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('./dien_timeline_inference_{}.json'.format(i), 'w') as f:
            f.write(chrome_trace)

    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / i

    return test_auc, accuracy_sum, eval_time


def inference(data_location,
              pb_path,
              batch_size=128,
              maxlen=100,
              data_type='fp32',
              graph_type='static',
              seed=2):

    print("graph location", pb_path)
    print("batch_size: ", batch_size)
    model_type = "DIEN"
    print("model: ", model_type)
    model_path = os.path.join(data_location, "dnn_save_path/ckpt_noshuff" + model_type + str(seed))
    best_model_path = os.path.join(data_location, "dnn_best_model/ckpt_noshuff" + model_type + str(seed))

    train_file = os.path.join(data_location, "local_train_splitByUser")
    test_file = os.path.join(data_location, "local_test_splitByUser")
    uid_voc = os.path.join(data_location, "uid_voc.pkl")
    mid_voc = os.path.join(data_location, "mid_voc.pkl")
    cat_voc = os.path.join(data_location, "cat_voc.pkl")

    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    input_layers = ["Inputs/uid_batch_ph",
                    "Inputs/mid_batch_ph",
                    "Inputs/cat_batch_ph",
                    "Inputs/mid_his_batch_ph",
                    "Inputs/cat_his_batch_ph",
                    "Inputs/mask",
                    "Inputs/target_ph"]
    if (graph_type == 'dynamic'):
      input_layers.append("Inputs/seq_len_ph")               
    input_tensor = [graph.get_tensor_by_name(x + ":0") for x in input_layers]
    output_layers = ["dien/fcn/add_6",
                     "dien/fcn/Metrics/Mean_1"]
    output_tensor = [graph.get_tensor_by_name(x + ":0") for x in output_layers]

    session_config = tf.compat.v1.ConfigProto()
    if data_type == 'bfloat16':
      graph_options = tf.compat.v1.GraphOptions(rewrite_options=rewriter_config_pb2.RewriterConfig(
                                              remapping=rewriter_config_pb2.RewriterConfig.AGGRESSIVE,
                                       auto_mixed_precision_mkl=rewriter_config_pb2.RewriterConfig.ON))
    if data_type == 'fp32':
       graph_options = tf.compat.v1.GraphOptions(rewrite_options=rewriter_config_pb2.RewriterConfig(
                                              remapping=rewriter_config_pb2.RewriterConfig.AGGRESSIVE))


    session_config = tf.compat.v1.ConfigProto(graph_options=graph_options)

    if args.num_intra_threads:
        session_config.intra_op_parallelism_threads = args.num_intra_threads
    if args.num_inter_threads:
        session_config.inter_op_parallelism_threads = args.num_inter_threads

    with tf.compat.v1.Session(graph=graph, config=session_config) as sess:
        test_data = DataIterator(data_location, test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)

        approximate_accelerator_time = 0

        total_data,num_iters = filtered_data(test_data) \
                               if args.exact_max_length <=0 \
                               else filtered_data(test_data, args.exact_max_length)
        test_auc, test_accuracy, eval_time= calculate(sess, total_data, input_tensor, output_tensor, batch_size)
        if args.accuracy_only :
           print('test_auc: %.4f ---- test_accuracy: %.9f ' % (test_auc, test_accuracy))
           return
       
        if args.exact_max_length:
          print("Exact Max length set to :",args.exact_max_length)
        else :
          print("Max length :100")
 
        #total_data, num_iters = filtered_data(test_data, args.exact_max_length)
        loop_iters = args.num_iterations 
        ignore_count = int(num_iters/20)
        if ignore_count == 0 and num_iters >= 5:
          ignore_count=1

        print("Info : First 5% batches (time) ignored as warm up:", ignore_count)
        for i in range(loop_iters):
            test_auc, test_accuracy, eval_time  = calculate(
                sess, total_data, input_tensor, output_tensor, batch_size, ignore_count)
            approximate_accelerator_time += eval_time
            print('Iter : %d test_auc: %.4f ---- test_accuracy: %.9f ---- eval_time: %.3f' % (i, test_auc, test_accuracy, eval_time))
        print("Batch_size ", batch_size)
        print("Batch count ", num_iters)
        print("Number of iterations", loop_iters)
        print("Total recommendations: %d" % (num_iters * batch_size))
        print("Approximate accelerator time in seconds is %.3f" % (approximate_accelerator_time/loop_iters))
        print("Approximate accelerator performance in recommendations/second is %.3f" %
              (float(loop_iters * (num_iters-ignore_count) * batch_size) / float(approximate_accelerator_time)))


if __name__ == '__main__':
    SEED = args.seed
    if tf.__version__[0] == '1':
        tf.compat.v1.set_random_seed(SEED)
    elif tf.__version__[0] == '2':
        tf.random.set_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)

    inference(data_location=args.data_location, seed=SEED, batch_size=args.batch_size,
              data_type=args.data_type, graph_type=args.graph_type, pb_path=args.input_graph)
