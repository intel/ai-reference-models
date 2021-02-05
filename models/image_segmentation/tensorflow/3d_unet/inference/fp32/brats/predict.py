import os

from train import config
from unet3d.prediction import run_validation_cases
from unet3d.prediction import run_large_batch_validation_cases

import argparse

parser = argparse.ArgumentParser(description='train opts:')
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--intra', '--num_intra_threads', type=int, default=56)
parser.add_argument('--inter', '--num_inter_threads', type=int, default=1)
parser.add_argument('--warmup', '--nw', type=int, default=10)
parser.add_argument('--report_interval', type=int, default=1)
parser.add_argument('--nb', type=int, default=10)

args = parser.parse_args()


import tensorflow as tf
from keras import backend as K
tf_config = tf.ConfigProto(intra_op_parallelism_threads=args.intra, inter_op_parallelism_threads=args.inter)
sess = tf.Session(graph=tf.get_default_graph(), config=tf_config)
K.set_session(sess)

def main():
    prediction_dir = os.path.abspath("prediction")

    # with tf.contrib.tfprof.ProfileContext('./profile_dir') as pctx:
    if args.bs == 1:
        run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir,
                         warmup=args.warmup,
                         report_interval=args.report_interval,
                         batch_size=args.bs,
                         n_batch=args.nb)
    else:
        run_large_batch_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir,
                         batch_size=args.bs,
                         report_interval=args.report_interval,
                         warmup=args.warmup,
                         n_batch=args.nb)


if __name__ == "__main__":
    main()
