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
# SPDX-License-Identifier: EPL-2.0
#

# -*- coding: utf-8 -*-

import os

LOCAL_T_MAX = 20 #20 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIRECTORY')
LOG_FILE = 'tmp/a3c_log'
ACCURACY_LOG_FILE = 'tmp/a3c_data'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 4 # parallel thread size // step increases by PARALLEL_SIZE
ROM = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pong.bin")     # action size = 3
ACTION_SIZE = 3 # action size

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_TIME_STEP = 8000 #10 * 10**7
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = False # To use GPU, set True
USE_LSTM = True # True for A3C LSTM, False for A3C FF
INFER_TIME_STEPS = 4000

TFPROFILE_ABSPATH_TRAIN='a3c/profile_train'
TFPROFILE_ABSPATH_INFER='a3c/profile_infer'