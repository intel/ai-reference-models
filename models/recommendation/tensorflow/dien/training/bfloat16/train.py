import pickle
import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *

import argparse

from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile

import os

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', help="mode, train or test")
parser.add_argument("--seed", type=int, default=3, help="seed value")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--data_type", type=str, default='fp32', help="data type: fp32 or fp16")
parser.add_argument("--num_intra_threads", type=int, default=None, help="num-intra-threads")
parser.add_argument("--num_inter_threads", type=int, default=None, help="num-inter-threads")
parser.add_argument("--data_location", type=str, default=None, help="data location")
parser.add_argument("--timeline", type=bool, default=False, help="obtain timeline")

args = parser.parse_args()

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0

TOTAL_TRAIN_SIZE = 512000


def prepare_data(input, target, maxlen=None, return_neg=False):
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
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    if args.data_type == 'fp32':
        data_type = 'float32'
    elif args.data_type == 'fp16':
        data_type = 'float16'
    elif args.data_type == 'bf16' or args.data_type=='bfloat16' :
        data_type = 'bfloat16'
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


def train(
    data_location,
    batch_size=128,
    maxlen=100,
    test_iter=100,
    save_iter=100,
    data_type='fp32',
    seed=2
):
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

    session_config = tf.compat.v1.ConfigProto()
    if args.num_intra_threads and args.num_inter_threads:
        session_config.intra_op_parallelism_threads = args.num_intra_threads
        session_config.inter_op_parallelism_threads = args.num_inter_threads

    with tf.compat.v1.Session(config=session_config) as sess:
        train_data = DataIterator(data_location, train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=False)
        test_data = DataIterator(data_location, test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()
        # Number of uid = 543060, mid = 367983, cat = 1601 for Amazon dataset
        print("Number of uid = %i, mid = %i, cat = %i" % (n_uid, n_mid, n_cat))
        model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type,
                                                batch_size=batch_size, max_length=maxlen, device='cpu')

        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        sys.stdout.flush()

        iter = 0
        lr = 0.001
        train_size = 0
        approximate_accelerator_time = 0
        total_accelerator_time = 0

        for itr in range(1):
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.

            if args.timeline:
                sample_freq = 200
                options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                run_metadata = tf.compat.v1.RunMetadata()

            total_data = []
            for src, tgt in train_data:
                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(
                    src, tgt, maxlen, return_neg=True)
                total_data.append([uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats])

            elapsed_time_records = []
            nums = 0
            for i in range(len(total_data)):
                nums += 1

                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = tuple(total_data[i])

                start_time = time.time()
                if args.timeline and nums == sample_freq:
                    loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr, noclk_mids, noclk_cats],
                                                      timeline_flag=True, options=options, run_metadata=run_metadata, step=nums)
                else:
                    loss, acc, aux_loss = model.train(
                        sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr, noclk_mids, noclk_cats])
                end_time = time.time()

                approximate_accelerator_time = end_time - start_time
                total_accelerator_time += approximate_accelerator_time
                elapsed_time_records.append(end_time - start_time)

                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iter += 1
                train_size += batch_size
                sys.stdout.flush()
                if (iter % test_iter) == 0:
                    # print("train_size: %d" % train_size)
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f' %
                          (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                    print("Approximate accelerator_time per batch: %.3f seconds" % approximate_accelerator_time)

                    # delete test every 100 iterations no need in training time
                    # print(' test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- eval_time: %.3f ---- num_iters: %d' % eval(sess, test_data, model, best_model_path))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % save_iter) == 0:
                    print('save model iter: %d' % (iter))
                    model.save(sess, model_path + "--" + str(iter))
                if train_size >= TOTAL_TRAIN_SIZE:
                    break

            print("iteration: ", nums)

            lr *= 0.5
            if train_size >= TOTAL_TRAIN_SIZE:
                break
        print("iter: %d" % iter)
        print("Total recommendations: %d" % TOTAL_TRAIN_SIZE)
        print("Toal accelerator time in seconds is %.3f" % total_accelerator_time)
        print("Approximate training performance in recommendations/second is %.3f" %
              (float(TOTAL_TRAIN_SIZE) / float(total_accelerator_time)))


if __name__ == '__main__':
    tf.compat.v1.disable_v2_behavior()
    SEED = args.seed
    if tf.__version__[0] == '1':
        tf.compat.v1.set_random_seed(SEED)
    elif tf.__version__[0] == '2':
        tf.random.set_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)

    #args.data_type='BF16'
    if args.mode == 'train':
        train(data_location=args.data_location, seed=SEED, batch_size=args.batch_size,
                data_type=args.data_type)
    else:
        print('do nothing...')
