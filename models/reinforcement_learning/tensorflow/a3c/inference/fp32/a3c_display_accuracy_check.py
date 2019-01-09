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
import logging
import os

from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
#from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier
from compare_inference_accuracy import compare_action_probs

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import CHECKPOINT_DIR
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import USE_LSTM
from constants import TFPROFILE_ABSPATH_INFER
from constants import LOG_FILE
from constants import ACCURACY_LOG_FILE
import tensorflow as tf
from argparse import ArgumentParser

arg_parser = ArgumentParser(description='The launchpad for all performance scripts.')
arg_parser.add_argument('-p',  help='do tfprofile', dest="do_tfprofile", default=False)
arg_parser.add_argument('-i', "--profile_absdir", help="The profile abs directory", dest="profile_path", default=TFPROFILE_ABSPATH_INFER)
arg_parser.add_argument('-ia', "--num_intra_threads", help='The intra size', type=int, dest="intra", default=44)
arg_parser.add_argument('-ie', "--num_inter_threads", help='The inter size', type=int, dest="inter", default=1)
arg_parser.add_argument('-nw', "--warmup", help='warmup number', type=int, dest="warmup", default=500)
arg_parser.add_argument('-nb', "--batches", help='number of batches', type=int, dest="numbatch", default=2000)
arg_parser.add_argument('-cp', "--ckpt", help='checkpoint', dest="ckpt_path", default=CHECKPOINT_DIR)
arg_parser.add_argument('-a', "--accuracy_log", help='accuracy log file', dest="accuracy_log", default=ACCURACY_LOG_FILE)
arg_parser.add_argument('-ag', "--accuracy_log_gpu", help='gpu accuracy log file', dest="accuracy_log_gpu", default = '')
intra = arg_parser.parse_args().intra
inter = arg_parser.parse_args().inter
warmup = arg_parser.parse_args().warmup
numbatch = arg_parser.parse_args().numbatch
accuracy_log = arg_parser.parse_args().accuracy_log

if USE_LSTM:
  accuracy_log += '_lstm'

# use CPU for display tool
device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"
  accuracy_log += '_gpu'
if os.path.isfile(accuracy_log):
  os.remove(accuracy_log)
logging.basicConfig(filename = accuracy_log, level = logging.INFO)

def choose_action(pi_values):
  return np.random.choice(range(len(pi_values)), p=pi_values)  

def run_infer(game_state, step_num):
  i = 0
  while (i<step_num):
    pi_values = global_network.run_policy(sess, game_state.s_t)

    action = choose_action(pi_values)
    game_state.process(action)
    #logging.info('step: {}, action: {}, reward: {}'.format(i, action, game_state.reward))
    logging.info('step: {}, pi_values: {}'.format(i, pi_values))
    if game_state.terminal:
      game_state.reset()
    else:
      game_state.update()
    i = i + 1

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
                        intra_op_parallelism_threads=intra,inter_op_parallelism_threads=inter)
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(arg_parser.parse_args().ckpt_path)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
else:
  print("Could not find old checkpoint")

game_state = GameState(0, display=False, no_op_max=0)
#below set random seed for random action selections
np.random.seed(0)

if arg_parser.parse_args().do_tfprofile:
  with tf.contrib.tfprof.ProfileContext(arg_parser.parse_args().profile_path,trace_steps=range(950, 1050, 1), dump_steps=[1050]) as pctx:
    run_infer(game_state, 1100)
else:
  run_infer(game_state, warmup) 
  run_infer(game_state, numbatch)
  graph = tf.get_default_graph()
  summary_writer = tf.summary.FileWriter(LOG_FILE, graph = graph)
  summary_writer.flush()
  summary_writer.close()

# if gpu log specified, then we'll compare with gpu log directly to check the accuracy correctness
gpu_log = arg_parser.parse_args().accuracy_log_gpu
# check if the default log existed
if not os.path.isfile(gpu_log):
  gpu_log = accuracy_log + '_gpu'
if USE_GPU:
  print('gpu log generated under: {}'.format(accuracy_log))
elif os.path.isfile(gpu_log):
  diff_sum, status = compare_action_probs(accuracy_log, gpu_log)
  print('Diff_sum: {}'.format(diff_sum)) 
  if status:
    print('Accuracy test of A3C: Pass')
  else:
    print('Accuracy test of A3C: Fail')
else:
  print('cpu log generated under: {}'.format(accuracy_log))
