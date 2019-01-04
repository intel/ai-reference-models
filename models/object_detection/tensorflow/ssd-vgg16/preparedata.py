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
#import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
# from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import cv2
#from utility import visualization
from nets.ssd import g_ssd_model
from preprocessing.ssd_vgg_preprocessing import np_image_unwhitened
from preprocessing.ssd_vgg_preprocessing import preprocess_for_train
from preprocessing.ssd_vgg_preprocessing import preprocess_for_eval
import tf_utils
import math



class PrepareData():
    def __init__(self):
        
        self.batch_size = 32
        self.labels_offset = 0
        
        self.data_format = 'NHWC'
        self.matched_thresholds = 0.5 #threshold for anchor matching strategy
      
        
       
        
        
        return
    def __preprocess_data(self, image, labels, bboxes):
        out_shape = g_ssd_model.img_shape
        if self.is_training_data:
            image, labels, bboxes = preprocess_for_train(image, labels, bboxes, out_shape = out_shape, data_format= self.data_format)
        else:
            image, labels, bboxes, _ = preprocess_for_eval(image, labels, bboxes, out_shape = out_shape, data_format= self.data_format)
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
                    common_queue_capacity=20 * self.batch_size,
                    common_queue_min=10 * self.batch_size)
        
        # Get for SSD network: image, labels, bboxes.
        [image, shape, format, filename, glabels, gbboxes,gdifficults] = provider.get(['image', 'shape', 'format','filename',
                                                         'object/label',
                                                         'object/bbox',
                                                         'object/difficult'])
        glabels -= self.labels_offset
        
        
        # Pre-processing image, labels and bboxes.
        image, glabels, gbboxes = self.__preprocess_data(image, glabels, gbboxes)

        # Assign groundtruth information for all default/anchor boxes
#         gclasses, glocalisations, gscores = g_ssd_model.tf_ssd_bboxes_encode(glabels, gbboxes)
        gclasses, glocalisations, gscores = g_ssd_model.match_achors(glabels, gbboxes, matching_threshold=self.matched_thresholds)
        
        
        return self.__batching_data(image, glabels, format, filename, gbboxes, gdifficults, gclasses, glocalisations, gscores)
    def __batching_data(self,image, glabels, format, filename, gbboxes, gdifficults,gclasses, glocalisations, gscores):
        
        #we will want to batch original glabels and gbboxes
        #this information is still useful even if they are padded after dequeuing
        dynamic_pad = True
        batch_shape = [1,1,1,1,1] + [len(gclasses), len(glocalisations), len(gscores)]
        tensors = [image, filename,glabels,gbboxes,gdifficults,gclasses, glocalisations, gscores]
        #Batch the samples
        if self.is_training_data:
            self.num_preprocessing_threads = 1
        else:
            # to make sure data is fectched in sequence during evaluation
            self.num_preprocessing_threads = 1
            
        #tf.train.batch accepts only list of tensors, this batch shape can used to
        #flatten the list in list, and later on convet it back to list in list.
        batch = tf.train.batch(
                tf_utils.reshape_list(tensors),
                batch_size=self.batch_size,
                num_threads=self.num_preprocessing_threads,
                dynamic_pad=dynamic_pad,
                capacity=5 * self.batch_size)
            
        #convert it back to the list in list format which allows us to easily use later on
        batch= tf_utils.reshape_list(batch, batch_shape)
        return batch
    def __disp_image(self, img, classes, bboxes):
        bvalid = (classes !=0)
        classes = classes[bvalid]
        bboxes = bboxes[bvalid]
        scores =np.full(classes.shape, 1.0)
        #visualization.plt_bboxes(img, classes, scores, bboxes,title='Ground Truth')
        return
    def __disp_matched_anchors(self,img, target_labels_data, target_localizations_data, target_scores_data):
        found_matched = False
        all_anchors = g_ssd_model.get_allanchors()
        for i, target_score_data in enumerate(target_scores_data):

            num_pos = (target_labels_data[i] != 0).sum()
            if (num_pos == 0):
                continue
            print('Found  {} matched default boxes in layer {}'.format(num_pos,g_ssd_model.feat_layers[i]))
#             pos_sample_inds = ((target_labels_data[i] != 0) & (target_score_data <=self.matched_thresholds)).nonzero()
            pos_sample_inds = (target_labels_data[i] != 0).nonzero()


            classes = target_labels_data[i][pos_sample_inds]
            scores = target_scores_data[i][pos_sample_inds]
            print("matched scores :{}".format(scores))
            print("matched labels: {}".format(classes))
            bboxes_default= g_ssd_model.get_allanchors(minmaxformat=True)[i][pos_sample_inds]
            
            
            
            bboxes_gt = g_ssd_model.decode_bboxes_layer(target_localizations_data[i][pos_sample_inds], 
                                       all_anchors[i][pos_sample_inds])
            
#             print("default box minimum, {} gt box minimum, {}".format(bboxes_default.min(), bboxes_gt.min()))
            
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
            #visualization.plt_bboxes(img, classes, scores, bboxes,neg_marks=neg_marks,title=title)
            found_matched = True  
            
        return found_matched
    def get_voc_2007_train_data(self,is_training_data=True):
        data_sources = "../data/voc/tfrecords/voc_train_2007*.tfrecord"
        num_samples = pascalvoc_datasets.DATASET_SIZE['2007_train']
       
        return self.__get_images_labels_bboxes(data_sources, num_samples, is_training_data)
    
    def get_voc_2012_train_data(self,is_training_data=True):
        data_sources = "../data/voc/tfrecords/voc_train_2012*.tfrecord"
        num_samples = pascalvoc_datasets.DATASET_SIZE['2012_train']
        
        return self.__get_images_labels_bboxes(data_sources, num_samples, is_training_data)
    
    def get_voc_2007_2012_train_data(self,is_training_data=True):
        data_sources = "../data/voc/tfrecords/voc_train*.tfrecord"
        num_samples = pascalvoc_datasets.DATASET_SIZE['2007_train'] + pascalvoc_datasets.DATASET_SIZE['2012_train']
        
        return self.__get_images_labels_bboxes(data_sources, num_samples, is_training_data)
    def get_voc_2007_test_data(self, data_location=None):
        if data_location is None:
          data_sources = "../data/voc/tfrecords/voc_test_2007*.tfrecord"
        else:
          data_sources = data_location+"/voc/tfrecords/voc_test_2007*.tfrecord"
        num_samples = pascalvoc_datasets.DATASET_SIZE['2007_test']
        
        return self.__get_images_labels_bboxes(data_sources, num_samples, False)
    
        
    def iterate_file_name(self, batch_data):
        
        num_batches = (int)(1*math.ceil(self.dataset.num_samples / float(self.batch_size)))
        print("num_batches %d" %num_batches)
        all_files = []
        with tf.Session('') as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            with slim.queues.QueueRunners(sess):
                for i in range(num_batches):
                    
                    image, filename,glabels,gbboxes,gdifficults,gclasses, glocalisations, gscores = sess.run(list(batch_data))
                    #print(filename)
                    all_files.append(filename)
                all_files = np.concatenate(all_files)
                
                
                all_files_unique = np.unique(all_files)
                print(len(all_files_unique))
            
        return
    def check_match_statistics(self,filename,gclasses, gscores):
        #flatten the array into Batch_num x bbox_num
        gt_anchor_labels = []
        gt_anchor_scores = []
        for i in range(len(gclasses)):
            gt_anchor_labels.append(np.reshape(gclasses[i], [self.batch_size, -1]))
            gt_anchor_scores.append(np.reshape(gscores[i], [self.batch_size,-1]))
        gt_anchor_labels = np.concatenate(gt_anchor_labels, axis = 1)
        gt_anchor_scores = np.concatenate(gt_anchor_scores, axis = 1)
        
        #find out missed match
        inds = (gt_anchor_scores <= self.matched_thresholds) & (gt_anchor_labels != 0)
        
        real_inds = inds.nonzero()

        print("missed match: {}".format(filename[real_inds[0]]))
        print("missed match scores: {}".format(gt_anchor_scores[real_inds]))
        print("missed match labels: {}".format(gt_anchor_labels[real_inds]))
        return
    def run(self):
        
        
         with tf.Graph().as_default():
#             batch_data= self.get_voc_2007_train_data(is_training_data=True)
            batch_data = self.get_voc_2007_test_data()
#             batch_data = self.get_voc_2012_train_data()
#            batch_data = self.get_voc_2007_2012_train_data(is_training_data = True)

#            print(batch_data)
            return self.iterate_file_name(batch_data)
            """           
            with tf.Session('') as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                with slim.queues.QueueRunners(sess):  
                    while True:  
                         
                        image, filename,glabels,gbboxes,gdifficults,gclasses, glocalisations, gscores = sess.run(list(batch_data))
                        
                        
                        
#                         print("min: {}, max: {}".format(gbboxes.min(), gbboxes.max()))
#                         return
                        
#                         print(glabels)
#                         print("number of zero label patch {}".format((glabels.sum(axis=1)  == 0).sum()))
#                         return
                       
#                         
                        
                         
                        print(filename)
                        selected_file =  b'000050'
                        picked_inds = None
                        #selet the first image in the batch
                        if selected_file is None:
                            picked_inds = 0
                        else:
                            picked_inds = (selected_file == filename).nonzero()
                            if len(picked_inds[0]) == 0:
                                picked_inds = None
                            else:
                                picked_inds = picked_inds[0][0]
                        
                        if picked_inds is None:
                            continue
                        
                        self.check_match_statistics(filename, gclasses, gscores)
                        target_labels_data = [item[picked_inds] for item in gclasses]
                        target_localizations_data = [item[picked_inds] for item in glocalisations]
                        target_scores_data = [item[picked_inds] for item in gscores]
                        image_data = image[picked_inds]
                        print("picked file {}".format(filename[picked_inds]))
 
                        image_data = np_image_unwhitened(image_data)
                        self.__disp_image(image_data, glabels[picked_inds], gbboxes[picked_inds])
                        found_matched = self.__disp_matched_anchors(image_data,target_labels_data, target_localizations_data, target_scores_data)
                        plt.show()
                        break;
                        #exit the batch data testing right after a successful match have been found
#                         if found_matched:
                        #this could be a potential issue to be solved since sometime not all grouth truth bboxes are encoded.
                        
                            
                        
        
        
               """
            return
    
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()
