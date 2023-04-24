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
# ==============================================================================

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def generate_data(input_shape, input_dtype="float32", batch_size=1):
    # np.random.seed(1024)
    if input_dtype in ["uint8", "int8", "int32", "int64"]:
        dummy_input = np.random.randint(1, size=input_shape).astype(input_dtype)
    else:
        dummy_input = np.random.randn(*input_shape).astype(input_dtype)
    return np.repeat(dummy_input[np.newaxis, :], batch_size, axis=0)


def preprocess_data(data_path):
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']
    raw_data = pd.read_csv(data_path + '/census-income.test.gz', header=None, names=column_names)
    raw_data['label_income'] = raw_data['income_50k'].map({' - 50000.': 0, ' 50000+.': 1})
    raw_data['label_marital'] = raw_data['marital_stat'].apply(lambda x: 1 if x == ' Never married' else 0)
    raw_data.drop(labels=['income_50k', 'marital_stat'], axis=1, inplace=True)

    columns = raw_data.columns.values.tolist()
    sparse_features = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                       'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                       'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                       'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                       'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                       'vet_question']
    dense_features = [col for col in columns if
                      col not in sparse_features and col not in ['label_income', 'label_marital']]

    raw_data[sparse_features] = raw_data[sparse_features].fillna('-1', )
    raw_data[dense_features] = raw_data[dense_features].fillna(0, )
    mms = MinMaxScaler(feature_range=(0, 1))
    raw_data[dense_features] = mms.fit_transform(raw_data[dense_features])

    for feat in sparse_features:
        lbe = LabelEncoder()
        raw_data[feat] = lbe.fit_transform(raw_data[feat])

    data = raw_data.drop(['label_income', 'label_marital'], axis=1)
    labels = raw_data[['label_income', 'label_marital']]

    return data, labels


def fetch_batch(infer_graph, batch_size, data, labels, i=0):
    input_dict = dict()
    if data is not None or labels is not None:
        for node in infer_graph.as_graph_def().node:
            if node.op == 'Placeholder':
                input_data = data[node.name].iloc[i * batch_size:(i + 1) * batch_size].to_numpy()
                input_data = input_data.reshape(input_data.shape[0], 1)
                input_dict[infer_graph.get_tensor_by_name(node.name + ":0")] = input_data
    else:
        for node in infer_graph.as_graph_def().node:
            if node.op == 'Placeholder':
                input_name = node.name
                input_dtype = tf.DType(node.attr['dtype'].type).name
                _shape = list(tf.TensorShape(node.attr['shape'].shape))
                input_shape = [item for item in _shape if item is not None]
                input_dict[infer_graph.get_tensor_by_name(input_name + ":0")] = generate_data(input_shape,
                                                                                              input_dtype, batch_size)
    return input_dict
