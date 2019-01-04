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



class CheckEncoding(object):
    def __init__(self):
        
        self.batch_size = 32
        
        
       
        
        
        return
    def __preprocess_data(self, image, labels, bboxes):
        out_shape = g_ssd_model.img_shape
        if self.is_training_data:
            image, labels, bboxes = preprocess_for_train(image, labels, bboxes, out_shape = out_shape)
        else:
            image, labels, bboxes, _ = preprocess_for_eval(image, labels, bboxes, out_shape = out_shape)
        return image, labels, bboxes
    def __get_images_labels_bboxes(self,data_sources, num_samples,is_training_data):
        
        self.dataset = pascalvoc_datasets.get_dataset_info(data_sources, num_samples)
        self.is_training_data = is_training_data
        if self.is_training_data:
            
            shuffle = True
            #make sure most samples can be fetched in one epoch
            self.num_readers = 2
        else:
            #make sure data is fetchd in sequence
            shuffle = False
            self.num_readers = 1
            
        
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    self.dataset,
                    shuffle=shuffle,
                    num_readers=self.num_readers,
                    common_queue_capacity=30 * self.batch_size,
                    common_queue_min=10 * self.batch_size)
        
        # Get for SSD network: image, labels, bboxes.
        [image, shape, format, self.filename, glabels, gbboxes,gdifficults] = provider.get(['image', 'shape', 'format','filename',
                                                         'object/label',
                                                         'object/bbox',
                                                         'object/difficult'])
        
        
        # Pre-processing image, labels and bboxes.
        self.image, self.glabels, self.gbboxes = self.__preprocess_data(image, glabels, gbboxes)
        
#         anchors_1 = g_ssd_model.get_allanchors(minmaxformat=False)
        anchors = g_ssd_model.get_allanchors(minmaxformat=True)
        print(anchors[-1][-4:])
        #flattent the anchors
        temp_anchors = []
        for i in range(len(anchors)):
            temp_anchors.append(tf.reshape(anchors[i], [-1, 4]))
        anchors = tf.concat(temp_anchors, axis=0)
        
        
        self.jaccard = g_ssd_model.compute_jaccard(self.gbboxes, anchors)

        # Assign groundtruth information for all default/anchor boxes
#         gclasses, glocalisations, gscores = g_ssd_model.tf_ssd_bboxes_encode(glabels, gbboxes)
        
        
        return
    
    def __disp_image(self, img, classes, bboxes):
        bvalid = (classes !=0)
        classes = classes[bvalid]
        bboxes = bboxes[bvalid]
        scores =np.full(classes.shape, 1.0)
        visualization.plt_bboxes(img, classes, scores, bboxes,title='Ground Truth')
        return
    def __disp_matched_anchors(self,img, target_labels_data, target_localizations_data, target_scores_data):
        found_matched = False
        all_anchors = g_ssd_model.get_allanchors()
        for i, target_score_data in enumerate(target_scores_data):

            num_pos = (target_score_data > 0.5).sum()
            if (num_pos == 0):
                continue
            print('Found  {} matched default boxes in layer {}'.format(num_pos,g_ssd_model.feat_layers[i]))
            pos_sample_inds = (target_score_data > 0.5).nonzero()
            pos_sample_inds = [pos_sample_inds[0],pos_sample_inds[1],pos_sample_inds[2]]

            classes = target_labels_data[i][pos_sample_inds]
            scores = target_scores_data[i][pos_sample_inds]
            bboxes_default= g_ssd_model.get_allanchors(minmaxformat=True)[i][pos_sample_inds]
            
            
            
            bboxes_gt = g_ssd_model.decode_bboxes_layer(target_localizations_data[i][pos_sample_inds], 
                                       all_anchors[i][pos_sample_inds])
            
            print("default box minimum, {} gt box minimum, {}".format(bboxes_default.min(), bboxes_gt.min()))
            
            marks_default = np.full(classes.shape, True)
            marks_gt = np.full(classes.shape, False)
            scores_gt = np.full(scores.shape, 1.0)
            
            bboxes = bboxes_default
            neg_marks = marks_default
            add_gt = True
            if add_gt :
                bboxes = np.vstack([bboxes_default,bboxes_gt])
                neg_marks = np.hstack([marks_default,marks_gt])
                classes = np.tile(classes, 2)
                scores = np.hstack([scores, scores_gt])
            
            title = "Default boxes: Layer {}".format(g_ssd_model.feat_layers[i])
            visualization.plt_bboxes(img, classes, scores, bboxes,neg_marks=neg_marks,title=title)
            found_matched = True  
            
        return found_matched
   
    def get_voc_2007_test_data(self):
        data_sources = "../../data/voc/tfrecords/voc_test_2007*.tfrecord"
        num_samples = pascalvoc_datasets.DATASET_SIZE['2007_test']
        
        return self.__get_images_labels_bboxes(data_sources, num_samples, False)
 
    def run(self):
        
        
        with tf.Graph().as_default():
            self.get_voc_2007_test_data()
            with tf.Session('') as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                with slim.queues.QueueRunners(sess):  
                     
                         
                    image, glabels, gbboxes,filename,jaccard= sess.run([self.image, self.glabels, self.gbboxes,self.filename,self.jaccard])
                    
                    print(filename)
                    print(glabels)
                    print(gbboxes)
                    print(jaccard)
                    
                     
                     
                    #selet the first image in the batch
                    
                    
                    image_data = np_image_unwhitened(image)
                    self.__disp_image(image_data, glabels, gbboxes)
                    #                         found_matched = self.__disp_matched_anchors(image_data,target_labels_data, target_localizations_data, target_scores_data)
                    plt.show()

                        
        
        
        
        return
    
    


if __name__ == "__main__":   
    obj= CheckEncoding()
    obj.run()