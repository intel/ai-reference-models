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


import tensorflow as tf
import numpy as np


class Detector(object):
    # net_factory:rnet or onet
    # datasize:24 or 48
    def __init__(self, net_factory, data_size, batch_size, model_path, num_inter_threads=0, num_intra_threads=0):
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 3], name='input_image')
            # figure out landmark
            self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True,
                                      inter_op_parallelism_threads=num_inter_threads,
                                      intra_op_parallelism_threads=num_intra_threads,
                                      gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            # check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert readstate, "the params dictionary is not valid"
            print("restore models' param")
            saver.restore(self.sess, model_path)

        self.data_size = data_size
        self.batch_size = batch_size
    # rnet and onet minibatch(test)

    def predict(self, databatch):
        # access data
        # databatch: N x 3 x data_size x data_size
        scores = []
        batch_size = self.batch_size

        minibatch = []
        cur = 0
        # num of all_data
        n = databatch.shape[0]
        while cur < n:
            # split mini-batch
            minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size
        # every batch prediction result
        cls_prob_list = []
        bbox_pred_list = []
        landmark_pred_list = []
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            # the last batch
            if m < batch_size:
                keep_inds = np.arange(m)
                # gap (difference)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
            # cls_prob batch*2
            # bbox_pred batch*4
            cls_prob, bbox_pred, landmark_pred = self.sess.run(
                [self.cls_prob, self.bbox_pred, self.landmark_pred], feed_dict={self.image_op: data})
            # num_batch * batch_size *2
            cls_prob_list.append(cls_prob[:real_size])
            # num_batch * batch_size *4
            bbox_pred_list.append(bbox_pred[:real_size])
            # num_batch * batch_size*10
            landmark_pred_list.append(landmark_pred[:real_size])
            # num_of_data*2,num_of_data*4,num_of_data*10
        return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)
