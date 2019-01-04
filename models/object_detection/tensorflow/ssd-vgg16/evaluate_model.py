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

import tensorflow as tf
#from compiler.ast import flatten
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import numpy as np
import math
from preparedata import PrepareData
from nets.ssd import g_ssd_model
import tf_extended as tfe
import time
from postprocessingdata import g_post_processing_data
import argparse

# For python3
def flatten(lst):
    """Flattens a list of lists"""
    return [subelem for elem in lst 
                    for subelem in elem]

class EvaluateModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        
        
        self.batch_size = 32
        self.labels_offset = 0
        self.eval_image_size = None
        self.preprocessing_name = None
        self.model_name = 'inception_v3'
        self.data_format = 'NHWC'
        self.num_preprocessing_threads = 4
        
        self.checkpoint_path =  None
        self.eval_dir = None
        
        
        return
    
    
    def __setup_eval(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        _ = slim.get_or_create_global_step()
        
        if self.eval_during_training:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
            
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        
        if self.eval_train:
            image, _, glabels,gbboxes,gdifficults, _, _, _ = self.get_voc_2007_train_data(is_training_data=False)
            self.eval_dir = './logs/evals/train_data'
        else:
            image, _, glabels,gbboxes,gdifficults, _, _, _ = self.get_voc_2007_test_data()
            self.eval_dir = './logs/evals/test_data'
            
        
       
        
        #get model outputs
        predictions, localisations, logits, end_points = g_ssd_model.get_model(image, data_format= self.data_format)
        
        
            
#         print_mAP_07_op, print_mAP_12_op = g_post_processing_data.get_mAP_tf_current_batch(predictions, localisations, glabels, gbboxes, gdifficults)
            
        names_to_updates = g_post_processing_data.get_mAP_tf_accumulative(predictions, localisations, glabels, gbboxes, gdifficults)
#         print_filename_op = tf.Print(filename, [filename], "input images: ")
        
        variables_to_restore = slim.get_variables_to_restore()
        
        num_batches = math.ceil(self.dataset.num_samples / float(self.batch_size))
        #num_batches=100
        
        config = tf.ConfigProto(log_device_placement=False,
                                gpu_options=gpu_options)
        
        
        if not self.eval_loop:
            # Standard evaluation loop.
            print("one time evaluate...")
            if tf.gfile.IsDirectory(self.checkpoint_path):
                checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_path)
            else:
                checkpoint_file = self.checkpoint_path
            tf.logging.info('Evaluating %s' % checkpoint_file)
            start = time.time()
            slim.evaluation.evaluate_once(
                master='',
                checkpoint_path=checkpoint_file,
                logdir=self.eval_dir,
                num_evals=num_batches,
                eval_op=flatten(list(names_to_updates.values())),
                session_config=config,
                variables_to_restore=variables_to_restore)
            # Log time spent.
            elapsed = time.time()
            elapsed = elapsed - start
            print('Time spent : %.3f seconds.' % elapsed)
            print('Time spent per BATCH: %.3f seconds.' % (elapsed / num_batches))
        else:
            print("evaluate during training...")
            # Waiting loop.
            slim.evaluation.evaluation_loop(
                master='',
                checkpoint_dir=self.checkpoint_path,
                logdir=self.eval_dir,
                num_evals=num_batches,
                eval_op=flatten(list(names_to_updates.values())),
                variables_to_restore=variables_to_restore,
                eval_interval_secs=60*60*2,
                session_config=config,
                max_number_of_evaluations=np.inf,
                timeout=None)
        
        

        
        
        
        
        return
    def parse_param(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', '--finetune',  help='whether use checkpoints under finetune folder',  action='store_true')
        parser.add_argument('-s', '--simul',  help='evaluate when training is onging',  action='store_true')
        parser.add_argument('-t', '--train',  help='evaluate aginst train dataset',  action='store_true')
        parser.add_argument('-l', '--loop',  help='evaluate checkpoints by loops',  action='store_true')
        parser.add_argument('-c', '--checkpoint',  help='evaluate a specific checkpoint',  default="")
        args = parser.parse_args()
        
        self.checkpoint_path = './logs/'
        self.finetune = args.finetune
        if args.finetune:
            self.checkpoint_path = './logs/finetune/'
        if args.checkpoint != "":
            self.checkpoint_path = args.checkpoint
            
        self.eval_during_training = args.simul
        self.eval_train = args.train
        self.eval_loop = args.loop
            
        return
    
    
    def run(self):
        self.parse_param()
        
        if self.eval_during_training:
            self.batch_size = 16
            #To evaluate while trainin going on
            with tf.device('/device:CPU:0'):      
                self.__setup_eval()
        else:
            self.__setup_eval()
                    
        
        
        return
    
    


if __name__ == "__main__":   
    obj= EvaluateModel()
    obj.run()
