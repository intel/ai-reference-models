#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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

# Census with Modin and Intel® Data Analytics and Acceleration Library (DAAL) Accelerated Scikit-Learn

# In this example we will be building an end to end machine learning workload with US census from 1970 to 2010.
# It uses Modin with Ray as compute engine for ETL, and uses Ridge Regression from daal accelerated scikit-learn library 
# to train and predict the US total income according to the education.

# Let's start by downloading census data to your local disk.
# Go to https://usa.ipums.org/usa-action/extract_requests/download and download ipums_education2income_1970-2010.csv.gz.
# You many need to register or login to your account to download.

# The data can also be downloader from here: https://rapidsai-data.s3.us-east-2.amazonaws.com/datasets/ipums_education2income_1970-2010.csv.gz

import os
import numpy as np

from sklearn import config_context
from sklearn.metrics import mean_squared_error, r2_score


# Import Modin and set Ray as the compute engine
import modin.pandas as pd
os.environ["MODIN_ENGINE"] = "ray"


# Load daal accelerated sklearn patch and import packages from the patch
import daal4py.sklearn
daal4py.sklearn.patch_sklearn()

from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm


# Read the data from the downloaded archive file
df = pd.read_csv('ipums_education2income_1970-2010.csv.gz', compression="gzip")

# ETL
# clean up unneeded features
keep_cols = [
    "YEAR", "DATANUM", "SERIAL", "CBSERIAL", "HHWT",
    "CPI99", "GQ", "PERNUM", "SEX", "AGE",
    "INCTOT", "EDUC", "EDUCD", "EDUC_HEAD", "EDUC_POP",
    "EDUC_MOM", "EDUCD_MOM2", "EDUCD_POP2", "INCTOT_MOM", "INCTOT_POP",
    "INCTOT_MOM2", "INCTOT_POP2", "INCTOT_HEAD", "SEX_HEAD",
]
df = df[keep_cols]

# clean up samples with invalid income, education, etc.
df = df.query("INCTOT != 9999999")
df = df.query("EDUC != -1")
df = df.query("EDUCD != -1")

# normalize income for inflation
df["INCTOT"] = df["INCTOT"] * df["CPI99"]

for column in keep_cols:
    df[column] = df[column].fillna(-1)
    df[column] = df[column].astype("float64")

y = df["EDUC"]
X = df.drop(columns=["EDUC", "CPI99"])


# Train the model and predict the income
# ML - training and inference
clf = lm.Ridge()

mse_values, cod_values = [], []
N_RUNS = 50
TRAIN_SIZE = 0.9
random_state = 777

X = np.ascontiguousarray(X, dtype=np.float64)
y = np.ascontiguousarray(y, dtype=np.float64)

# cross validation
for i in range(N_RUNS):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE,
                                                        random_state=random_state)
    random_state += 777

    # training
    with config_context(assume_finite=True):
        model = clf.fit(X_train, y_train)

    # inference
    y_pred = model.predict(X_test)

    mse_values.append(mean_squared_error(y_test, y_pred))
    cod_values.append(r2_score(y_test, y_pred))


# Check the regression results: mean squared error and r square score
mean_mse = sum(mse_values)/len(mse_values)
mean_cod = sum(cod_values)/len(cod_values)
mse_dev = pow(sum([(mse_value - mean_mse)**2 for mse_value in mse_values])/(len(mse_values) - 1), 0.5)
cod_dev = pow(sum([(cod_value - mean_cod)**2 for cod_value in cod_values])/(len(cod_values) - 1), 0.5)
print("mean MSE ± deviation: {:.9f} ± {:.9f}".format(mean_mse, mse_dev))
print("mean COD ± deviation: {:.9f} ± {:.9f}".format(mean_cod, cod_dev))


# Here are our scores:
# ```
# mean MSE ± deviation: 0.032564569 ± 0.000041799
# mean COD ± deviation: 0.995367533 ± 0.000005869
# ```
# 
