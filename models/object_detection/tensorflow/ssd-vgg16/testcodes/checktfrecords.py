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

from datasets import dataset_factory
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
# from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np

from utility import visualization

import tf_utils



class CheckTfrecords():
    def __init__(self):
        
        return
    def __get_images_labels_bboxes(self):
        dataset = dataset_factory.get_dataset(
                self.dataset_name, self.dataset_split_name, self.dataset_dir)

        #make sure data is fetchd in sequence
        shuffle = False
        self.num_readers = 1
            
        
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    shuffle=shuffle,
                    num_readers=self.num_readers,
                    common_queue_capacity=30 * self.batch_size,
                    common_queue_min=10 * self.batch_size)
        
        # Get for SSD network: image, labels, bboxes.
        [image, shape, format, filename, glabels, gbboxes,gdifficults] = provider.get(['image', 'shape', 'format','filename',
                                                         'object/label',
                                                         'object/bbox',
                                                         'object/difficult'])
      
        
        
        
        return image, shape, format, filename, glabels, gbboxes,gdifficults
    
    
    def run(self):
        
        self.dataset_name = 'pascalvoc_2007'
        self.dataset_split_name = 'train'
        self.dataset_dir = '../../data/voc/tfrecords/'
        self.batch_size = 32
        
        with tf.Graph().as_default():
            tensors = self.__get_images_labels_bboxes()
            with tf.Session('') as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                with slim.queues.QueueRunners(sess):  
                    for i in range(5000):
                        
                           
                            image, shape, format, filename, glabels, gbboxes,gdifficults = sess.run(list(tensors))
                            
                            if str(filename,'utf-8') == "000394":
                                print(str(filename,'utf-8'))
                                scores = np.full(glabels.shape, 1.0)
                                visualization.plt_bboxes(image, glabels, scores, gbboxes,title=filename)
                                plt.show()
                                
                                
                                break
                            
                        
        
        
        
        return
    
    


if __name__ == "__main__":   
    obj= CheckTfrecords()
    obj.run()