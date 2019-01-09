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
#import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time

from a3c_training_thread import A3CTrainingThread
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
#from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
#from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import USE_LSTM
import tensorflow as tf

max_time_step = 2000
#tf.logging.set_verbosity(tf.logging.INFO)

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

global_t = 0
global_process = 0

stop_requested = False

if USE_LSTM:
  global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)
else:
  global_network = GameACFFNetwork(ACTION_SIZE, -1, device)


training_threads = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

for i in range(1):
  training_thread = A3CTrainingThread(i, global_network, initial_learning_rate,
                                      learning_rate_input,
                                      grad_applier, max_time_step,
                                      device = device)
  training_threads.append(training_thread)

# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True,
                                        intra_op_parallelism_threads=int(56/PARALLEL_SIZE/2), inter_op_parallelism_threads=2))
#                                        intra_op_parallelism_threads=int(44/2), inter_op_parallelism_threads=2))
#                                        intra_op_parallelism_threads=int(56), inter_op_parallelism_threads=1))

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
score_input = tf.placeholder(tf.int32)
tf.summary.scalar("score", score_input)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

# init or load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print(">>> global step set: ", global_t)
  # set wall time
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'r') as f:
    wall_t = float(f.read())
else:
  print("Could not find old checkpoint")
  # set wall time
  wall_t = 0.0


def train_function(parallel_index):
  global global_t
  global global_process
  
  training_thread = training_threads[parallel_index]
  # set start_time
  start_time = time.time() - wall_t
  print("start_time:", start_time)
  training_thread.set_start_time(start_time)
  with tf.contrib.tfprof.ProfileContext('/home/tianleli/tmp/profile'+str(parallel_index),trace_steps=range(50, 100, 1), dump_steps=[101]) as pctx:
#  if True:
    while True:
      if stop_requested:
        break
      if global_t > max_time_step:
        break

      diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                         summary_op, score_input)
      global_t += diff_global_t
      #if parallel_index == 0:
      #  global_process = global_process+1
    
def signal_handler(signal, frame):
  global stop_requested
  print("You pressed Ctrl+C")
  stop_requested = True
  
train_threads = []
for i in range(1):
  train_threads.append(threading.Thread(target=train_function, args=(i,)))
  
signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t

for t in train_threads:
  print("t.start", t)
  t.start()

  #print('Press Ctrl+C to stop')
  #signal.pause()

  #print('Now saving data. Please wait')
  
for t in train_threads:
  print("t.join",t)
  t.join()
#elapsed_time = time.time() - start_time
#print("end time: {%.2f} seconds"%elapsed_time )
#print("steps/second: {%.2f}"%(max_time_step/elapsed_time))
#print("global_process:", global_process)
if not os.path.exists(CHECKPOINT_DIR):
  os.mkdir(CHECKPOINT_DIR)  

# write wall time
wall_t = time.time() - start_time
wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
with open(wall_t_fname, 'w') as f:
  f.write(str(wall_t))

saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)

