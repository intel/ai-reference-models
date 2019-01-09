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
import numpy as np
import random
import time
from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
#from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import CHECKPOINT_DIR
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import USE_LSTM
from constants import INFER_TIME_STEPS
import tensorflow as tf

def choose_action(pi_values):
  return np.random.choice(range(len(pi_values)), p=pi_values)  

# use CPU for display tool
device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

if USE_LSTM:
  global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)
else:
  global_network = GameACFFNetwork(ACTION_SIZE, -1, device)

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=int(28),inter_op_parallelism_threads=1)
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
else:
  print("Could not find old checkpoint")

game_state = GameState(0, display=False, no_op_max=0)

#start_time = time.time()
i = 0
with tf.contrib.tfprof.ProfileContext('/home/tianleli/tmp/profile',trace_steps=range(50, 100, 1), dump_steps=[101]) as pctx:
#if True:
  while (i<200): #INFER_TIME_STEPS):
  #while True:
    pi_values = global_network.run_policy(sess, game_state.s_t)

    action = choose_action(pi_values)
    game_state.process(action)
    if game_state.terminal:
      game_state.reset()
    else:
      game_state.update()
    i = i+1
#infer_time = time.time()-start_time
#print("end:{%f} seconds"%(infer_time))
#print("infer steps/second: {%f}"%(INFER_TIME_STEPS/infer_time))
