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

from datasets import pascalvoc_datasets
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
# from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import cv2
from utility import visualization
from nets.ssd import g_ssd_model
from preprocessing.ssd_vgg_preprocessing import np_image_unwhitened
from preprocessing.ssd_vgg_preprocessing import preprocess_for_train
from preprocessing.ssd_vgg_preprocessing import preprocess_for_eval
import tf_utils
import math
from preparedata import PrepareData


class CheckImages(PrepareData):
    def __init__(self):
        
        PrepareData.__init__(self)
        return
    
    def list_images(self,sess, batch_data,target_object=1):
        i = 0
        num_batches = math.ceil(self.dataset.num_samples / float(self.batch_size))
        target_filenames = []
        num_bboxes = 0
        while i < num_batches:  
             
            image, filename,glabels,gbboxes,gdifficults,gclasses, glocalisations, gscores = sess.run(list(batch_data))
            pos_sample_inds = (glabels == target_object).nonzero()
            num_bboxes += len(pos_sample_inds[0])
            target_filenames.extend(list(filename[np.unique(pos_sample_inds[0])]))
           
            i += 1
#         print("number of matched image {} matched bboxes {} for {}, \n{}".format(len(target_filenames), num_bboxes, target_object, np.array(target_filenames)))
        print("{}, number of matched image {} matched bboxes {}, ratio {}".format(target_object, len(target_filenames), num_bboxes, float(num_bboxes)/len(target_filenames)))
        return
    
    
    def run(self):
        
        
        with tf.Graph().as_default():
#             batch_data= self.get_voc_2007_train_data(is_training_data=False)
#             batch_data = self.get_voc_2007_test_data()
#             batch_data = self.get_voc_2012_train_data()
            batch_data = self.get_voc_2007_2012_train_data(is_training_data = False)


#             return self.iterate_file_name(batch_data)
           
            with tf.Session('') as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                with slim.queues.QueueRunners(sess):  
#                     target_object = 9
                    for target_object in np.arange(1,21):
                        self.list_images(sess, batch_data,target_object)
                    
                            
                        
        
        
        
        return
    
    


if __name__ == "__main__":   
    obj= CheckImages()
    obj.run()