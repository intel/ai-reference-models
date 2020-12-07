# Copyright (c) 2020 Intel Corporation
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
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_rcv1
import timeit
import os

t0 = timeit.default_timer()

rcv1 = fetch_rcv1()

rcv1_data = rcv1.data
rcv1_target = rcv1.target

t1 = timeit.default_timer()
time_load = t1 - t0

t0 = timeit.default_timer()

x_train, x_test, y_train, y_test = train_test_split(
    rcv1_data, rcv1_target, test_size=0.05, random_state=42)

from daal4py.sklearn import patch_sklearn
patch_sklearn()

from sklearn.svm import SVC

t1 = timeit.default_timer()
time_train_test_split = t1 - t0

print('[Data] train: {} test: {}'.format(x_train.shape, x_test.shape))
print('[Target] train: {} test: {}'.format(y_train.shape, y_test.shape))


print('[Time] Load time {} sec'.format(time_load))
print('[Time] train_test_split time {} sec'.format(time_train_test_split))

t0 = timeit.default_timer()

clf = SVC(C=100.0, kernel='rbf', cache_size=8*1024)
svm = OneVsRestClassifier(clf, n_jobs=4)
svm.fit(x_train, y_train)

t1 = timeit.default_timer()
time_fit_train_run = t1 - t0

print('[Time] Fit time {} sec'.format(time_fit_train_run))

t0 = timeit.default_timer()
svm_prediction = svm.predict(x_test)
t1 = timeit.default_timer()
time_predict_test_run = t1 - t0

print('[Time] Predict time {} sec'.format(time_predict_test_run))

t0 = timeit.default_timer()
print('Accuracy score is {}'.format(accuracy_score(y_test, svm_prediction)))
print('F1 samples score is {}'.format(
    f1_score(y_test, svm_prediction, average='samples')))
print('F1 micro score is {}'.format(
    f1_score(y_test, svm_prediction, average='micro')))
t1 = timeit.default_timer()
time_metric_run = t1 - t0

print('[Time] Metric time {} sec'.format(time_metric_run))
