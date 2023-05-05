#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
'''This script used to create and train the MMoE model on the UCI Census Income Dataset'''

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import MMOE

from argparse import ArgumentParser
import tensorflow as tf
import time


class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.times = []
        self.throughput = []

    def on_batch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_batch_end(self, batch, logs={}):
        total_time = time.time() - self.epoch_time_start
        self.times.append(total_time)
        self.throughput.append(self.batch_size / total_time)


def train_and_eval():
    print("Training with " + str(args.precision) + " precision and batch size of " + str(args.batch_size))

    if args.precision == "bfloat16":
        print("Enabling auto-mixed precision for bfloat16")
        tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16': True})
        print(tf.config.optimizer.get_experimental_options())
    elif args.precision == "fp16":
        print("Enabling auto-mixed precision for fp16")
        tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
        print(tf.config.optimizer.get_experimental_options())

    # Read the training data
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']
    data = pd.read_csv(args.data_location + '/census-income.data.gz', header=None, names=column_names)

    # Data pre-processing and normalization
    data['label_income'] = data['income_50k'].map({' - 50000.': 0, ' 50000+.': 1})
    data['label_marital_stat'] = data['marital_stat'].apply(lambda x: 1 if x == ' Never married' else 0)
    data.drop(labels=['income_50k', 'marital_stat'], axis=1, inplace=True)

    columns = data.columns.values.tolist()
    sparse_features = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                       'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                       'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                       'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                       'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                       'vet_question']
    dense_features = [col for col in columns if
                      col not in sparse_features and col not in ['label_income', 'label_marital_stat']]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4) for feat in sparse_features] + \
                             [DenseFeat(feat, 1, ) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # Generate train and val data for the model
    train, val = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}

    # Define the model. Train, predict and evaluate
    model = MMOE(dnn_feature_columns, expert_dnn_hidden_units=(64, 8), tower_dnn_hidden_units=[],
                 task_types=['binary', 'binary'], task_names=['label_income', 'label_marital_stat'])
    model.compile(optimizer='adam', loss=["binary_crossentropy", "binary_crossentropy"],
                  metrics=['binary_crossentropy'], )

    time_callback = TimeHistory(batch_size=args.batch_size)
    model.fit(train_model_input, [train['label_income'].values, train['label_marital_stat'].values],
              batch_size=args.batch_size, epochs=args.train_epochs, verbose=2, validation_split=0.2,
              callbacks=[time_callback])
    avg_throughput = sum(time_callback.throughput) / len(time_callback.throughput)
    print("Average Throughput: " + str(round(avg_throughput, 2)) + " examples/sec\n")

    # Saving the model
    model.save(args.output_dir)
    print("The trained model is saved at: " + str(args.output_dir) + "\n")

    # Running predictions on the validation dataset
    pred_ans = model.predict(val_model_input)

    print("*** Results on the Validation Dataset ***")
    print("Income AUC: ", round(roc_auc_score(val['label_income'], pred_ans[0]), 4))
    print("Marital Status AUC: ", round(roc_auc_score(val['label_marital_stat'], pred_ans[1]), 4))


def get_arg_parser():
    arg_parser = ArgumentParser(description='Parse args')
    arg_parser.add_argument(
        '-d', '--data-location',
        help='Specify the location of the training data',
        dest='data_location', required=True
    )
    arg_parser.add_argument(
        '-e', '--train-epochs',
        help='Specify the number of epochs for training',
        type=int, dest='train_epochs',
        default=25
    )
    arg_parser.add_argument(
        '-b', '--batch-size',
        help='Batch size for training',
        type=int, dest='batch_size',
        default=256
    )
    arg_parser.add_argument(
        '-p', '--precision',
        help='Specify the model precision to use: fp32, bfloat16 or fp16',
        choices=['fp32', 'bfloat16', 'fp16'],
        dest="precision", required=True
    )
    arg_parser.add_argument(
        '-o', '--output-dir',
        help='Specify the location of the output directory for logs and saved model',
        dest='output_dir', required=True
    )
    return arg_parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    train_and_eval()
