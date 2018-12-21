#
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
# SPDX-License-Identifier: EPL-2.0
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim

from preparedata import PrepareData
from nets.ssd import g_ssd_model
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from postprocessingdata import g_post_processing_data
from tensorflow.python.training import saver as tf_saver


class TrainModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        
        self.num_epochs_per_decay = 2.0
        self.learning_rate_decay_type = 'exponential'
        self.end_learning_rate =  0.0001
        self.learning_rate = 0.01
        
        #optimiser
        self.optimizer = 'rmsprop'
        
        self.data_format = 'NHWC'
        self.adadelta_rho = 0.95
        self.opt_epsilon= 1.0
        self.adagrad_initial_accumulator_value= 0.1
        self.adam_beta1= 0.9
        self.adam_beta2= 0.999
        self.ftrl_learning_rate_power = -0.5
        self.ftrl_initial_accumulator_value = 0.1
        self.ftrl_l1= 0.0
        self.ftrl_l2 = 0.0
        self.momentum= 0.9
        
        self.rmsprop_decay = 0.9
        self.rmsprop_momentum = 0.9
        
        self.train_dir = '/tmp/tfmodel/'
        self.max_number_of_steps = None

        
        self.checkpoint_path = None
        self.checkpoint_exclude_scopes = None
        self.ignore_missing_vars = False
        
        self.batch_size= 32
        
        self.save_interval_secs = 60*60#one hour
        self.save_summaries_secs= 60

        
        
        
        
        self.label_smoothing = 0
        return
    
    def __configure_learning_rate(self, num_samples_per_epoch, global_step):
        """Configures the learning rate.
    
        Args:
            num_samples_per_epoch: The number of samples in each epoch of training.
            global_step: The global_step tensor.
    
        Returns:
            A `Tensor` representing the learning rate.
    
        Raises:
            ValueError: if
        """
        decay_steps = int(num_samples_per_epoch / self.batch_size *
                                            self.num_epochs_per_decay)
       
    
        if self.learning_rate_decay_type == 'exponential':
            return tf.train.exponential_decay(self.learning_rate,
                                                                                global_step,
                                                                                decay_steps,
                                                                                self.learning_rate_decay_factor,
                                                                                staircase=True,
                                                                                name='exponential_decay_learning_rate')
        elif self.learning_rate_decay_type == 'fixed':
            return tf.constant(self.learning_rate, name='fixed_learning_rate')
        elif self.learning_rate_decay_type == 'polynomial':
            return tf.train.polynomial_decay(self.learning_rate,
                                                                             global_step,
                                                                             decay_steps,
                                                                             self.end_learning_rate,
                                                                             power=1.0,
                                                                             cycle=False,
                                                                             name='polynomial_decay_learning_rate')
        else:
            raise ValueError('learning_rate_decay_type [%s] was not recognized',
                                         self.learning_rate_decay_type)
        return
    def __configure_optimizer(self, learning_rate):
        """Configures the optimizer used for training.
    
        Args:
            learning_rate: A scalar or `Tensor` learning rate.
    
        Returns:
            An instance of an optimizer.
    
        Raises:
            ValueError: if FLAGS.optimizer is not recognized.
        """
        if self.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate,
                    rho=self.adadelta_rho,
                    epsilon=self.opt_epsilon)
        elif self.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                    learning_rate,
                    initial_accumulator_value=self.adagrad_initial_accumulator_value)
        elif self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                    learning_rate,
                    beta1=self.adam_beta1,
                    beta2=self.adam_beta2,
                    epsilon=self.opt_epsilon)
        elif self.optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                    learning_rate,
                    learning_rate_power=self.ftrl_learning_rate_power,
                    initial_accumulator_value=self.ftrl_initial_accumulator_value,
                    l1_regularization_strength=self.ftrl_l1,
                    l2_regularization_strength=self.ftrl_l2)
        elif self.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                    learning_rate,
                    momentum=self.momentum,
                    name='Momentum')
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                    learning_rate,
                    decay=self.rmsprop_decay,
                    momentum=self.rmsprop_momentum,
                    epsilon=self.opt_epsilon)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer [%s] was not recognized', self.optimizer)
        return optimizer
    def __get_variables_to_train(self):
        """Returns a list of variables to train.
    
        Returns:
            A list of variables to train by the optimizer.
        """
        if self.trainable_scopes is None:
            return tf.trainable_variables()
        else:
            scopes = [scope.strip() for scope in self.trainable_scopes.split(',')]
    
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train
    
    
    def __start_training(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        
        #get batched training training data 
        image, filename,glabels,gbboxes,gdifficults,gclasses, localizations, gscores = self.get_voc_2007_2012_train_data()
        
        #get model outputs
        predictions, localisations, logits, end_points = g_ssd_model.get_model(image, weight_decay=self.weight_decay, is_training=True, data_format= self.data_format)
        
        #get model training losss
        total_loss = g_ssd_model.get_losses(logits, localisations, gclasses, localizations, gscores)

        
        
        global_step = slim.create_global_step()
        
        # Variables to train.
        variables_to_train = self.__get_variables_to_train()
        
        learning_rate = self.__configure_learning_rate(self.dataset.num_samples, global_step)
        optimizer = self.__configure_optimizer(learning_rate)
        
        
        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=variables_to_train)
        
        self.__add_summaries(end_points, learning_rate, total_loss)
        
        self.setup_debugging(predictions, localizations, glabels, gbboxes, gdifficults)
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(log_device_placement=False,
                                gpu_options=gpu_options)
        
        ###########################
        # Kicks off the training. #
        ###########################
       
        slim.learning.train(
                train_op,
                self.train_dir,
                train_step_fn=self.train_step,
                saver=tf_saver.Saver(max_to_keep=500),
                init_fn=self.__get_init_fn(),
                number_of_steps=self.max_number_of_steps,
                log_every_n_steps=self.log_every_n_steps,
                save_summaries_secs=self.save_summaries_secs,
#                 session_config=config,
                save_interval_secs=self.save_interval_secs)
        
        
        return
    def setup_debugging(self,predictions, localizations, glabels, gbboxes, gdifficults):
#         image_eval, filename_eval,glabels_eval,gbboxes_eval,gdifficults_eval,gclasses_eval, localizations_eval, gscores_eval = self.get_voc_2007_test_data()
#         predictions_eval, localisations_eval, _, _ = g_ssd_model.get_model(image_eval, weight_decay=self.weight_decay)
#         _, self.mAP_12_op_eval = g_post_processing_data.get_mAP_tf_current_batch(predictions_eval, localizations_eval, glabels_eval, gbboxes_eval, gdifficults_eval)
        
        _, self.mAP_12_op_train = g_post_processing_data.get_mAP_tf_current_batch(predictions, localizations, glabels, gbboxes, gdifficults)
        return
    def debug_training(self,sess,global_step):
        np_global_step = sess.run(global_step)
        if np_global_step % self.log_every_n_steps != 0:
            return
        
       
        m_AP_12 = sess.run(self.mAP_12_op_train)
        logging.info("step {}/{}, m_AP_12 {}".format(np_global_step, self.max_number_of_steps, m_AP_12))
        
        return
    def train_step(self, sess, train_op, global_step, train_step_kwargs):
        """Function that takes a gradient step and specifies whether to stop.
    
        Args:
            sess: The current session.
            train_op: An `Operation` that evaluates the gradients and returns the
                total loss.
            global_step: A `Tensor` representing the global training step.
            train_step_kwargs: A dictionary of keyword arguments.
    
        Returns:
            The total loss and a boolean indicating whether or not to stop training.
    
        Raises:
            ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
        """
        
    
        start_time = time.time()
        trace_run_options = None
        run_metadata = None
        if 'should_trace' in train_step_kwargs:
            if 'logdir' not in train_step_kwargs:
                raise ValueError('logdir must be present in train_step_kwargs when '
                                                 'should_trace is present')
            if sess.run(train_step_kwargs['should_trace']):
                trace_run_options = config_pb2.RunOptions(
                        trace_level=config_pb2.RunOptions.FULL_TRACE)
                run_metadata = config_pb2.RunMetadata()
    
        total_loss, np_global_step = sess.run([train_op, global_step],
                                                                                    options=trace_run_options,
                                                                                    run_metadata=run_metadata)
        time_elapsed = time.time() - start_time
        
#         self.debug_training(sess,global_step)
        
    
        if run_metadata is not None:
            tl = timeline.Timeline(run_metadata.step_stats)
            trace = tl.generate_chrome_trace_format()
            trace_filename = os.path.join(train_step_kwargs['logdir'],
                                                                        'tf_trace-%d.json' % np_global_step)
            logging.info('Writing trace to %s', trace_filename)
            file_io.write_string_to_file(trace_filename, trace)
            if 'summary_writer' in train_step_kwargs:
                train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                                                                                         'run_metadata-%d' %
                                                                                                                         np_global_step)
    
        if 'should_log' in train_step_kwargs:
            if sess.run(train_step_kwargs['should_log']):
                logging.info('global step %d: loss = %.4f (%.2f sec/step)',
                                         np_global_step, total_loss, time_elapsed)
    
        # TODO(nsilberman): figure out why we can't put this into sess.run. The
        # issue right now is that the stop check depends on the global step. The
        # increment of global step often happens via the train op, which used
        # created using optimizer.apply_gradients.
        #
        # Since running `train_op` causes the global step to be incremented, one
        # would expected that using a control dependency would allow the
        # should_stop check to be run in the same session.run call:
        #
        #     with ops.control_dependencies([train_op]):
        #         should_stop_op = ...
        #
        # However, this actually seems not to work on certain platforms.
        if 'should_stop' in train_step_kwargs:
            should_stop = sess.run(train_step_kwargs['should_stop'])
        else:
            should_stop = False
    
        return total_loss, should_stop
    def __add_summaries(self,end_points,learning_rate,total_loss):
        # Add summaries for end_points (activations).

        for end_point in end_points:
            x = end_points[end_point]
            tf.summary.histogram('activations/' + end_point, x)
            tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x))
        # Add summaries for losses and extra losses.
        
        tf.summary.scalar('total_loss', total_loss)
        for loss in tf.get_collection('EXTRA_LOSSES'):
            tf.summary.scalar(loss.op.name, loss)

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

        return
    def __get_init_fn(self):
        """Returns a function run by the chief worker to warm-start the training.
    
        Note that the init_fn is only run when initializing the model during the very
        first global step.
    
        Returns:
            An init function run by the supervisor.
        """  
        
        if self.checkpoint_path is None:
            return None
    
        # Warn the user if a checkpoint exists in the train_dir. Then we'll be
        # ignoring the checkpoint anyway.
        
        
        if tf.train.latest_checkpoint(self.train_dir):
            tf.logging.info(
                    'Ignoring --checkpoint_path because a checkpoint already exists in %s'
                    % self.train_dir)
            return None
    
        exclusions = []
        if self.checkpoint_exclude_scopes:
            exclusions = [scope.strip()
                                        for scope in self.checkpoint_exclude_scopes.split(',')]
    
        # TODO(sguada) variables.filter_variables()
        variables_to_restore = []
        all_variables = slim.get_model_variables()
        if self.fine_tune_vgg16:
            global_step = slim.get_or_create_global_step()
            all_variables.append(global_step)
        for var in all_variables:
            excluded = False
            
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
    
        if tf.gfile.IsDirectory(self.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        else:
            checkpoint_path = self.checkpoint_path
    
        tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    
        return slim.assign_from_checkpoint_fn(
                checkpoint_path,
                variables_to_restore,
                ignore_missing_vars=self.ignore_missing_vars)
    
    def run(self):
        
        #fine tune the new parameters
        self.train_dir = './logs'
        
        
        self.checkpoint_path = '../data/trained_models/vgg16/vgg_16.ckpt'
        self.checkpoint_exclude_scopes = g_ssd_model.model_name
        self.trainable_scopes = g_ssd_model.model_name
        
        
        self.max_number_of_steps = 30000
        self.log_every_n_steps = 10
        
        self.learning_rate = 0.1
        self.learning_rate_decay_type = 'fixed'
        
        
        self.optimizer = 'adam'
        self.weight_decay = 0.0005 # for model regularization
        
        self.fine_tune_vgg16 = True
        
        if self.fine_tune_vgg16:  
            #fine tune all parameters
            self.train_dir = './logs/finetune'
            self.checkpoint_path =  './logs'
            self.checkpoint_exclude_scopes = None
            self.trainable_scopes = "{},vgg_16".format(g_ssd_model.model_name)
            self.max_number_of_steps = 120000
            self.learning_rate=0.0005

       
        
        
        self.__start_training()
        return
    
    


if __name__ == "__main__":   
    obj= TrainModel()
    obj.run()