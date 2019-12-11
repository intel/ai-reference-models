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

#

# coding:utf-8
import sys

from MtcnnDetector import MtcnnDetector
from detector import Detector
from fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
import cv2
import time
import os
import numpy as np

import tensorflow as tf

flags = tf.flags

# opmization parameters
flags.DEFINE_integer('num_intra_threads', 0,
                     'Specifiy the number threads within layers')
flags.DEFINE_integer('num_inter_threads', 0,
                     'Specify the number threads between layers')
flags.DEFINE_string('dl', None, 'Location of data.')
flags.DEFINE_string('ckpt', None,
                    'Directory where the model was written to.')

FLAGS = flags.FLAGS

print(FLAGS.num_inter_threads)
print(FLAGS.num_intra_threads)
print(FLAGS.ckpt)


test_mode = "ONet"
thresh = [0.9, 0.6, 0.7]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = [FLAGS.ckpt + '/PNet_landmark/PNet', FLAGS.ckpt + '/RNet_landmark/RNet', FLAGS.ckpt + '/ONet_landmark/ONet']

epoch = [18, 14, 16]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
# load pnet model
if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0], FLAGS.num_inter_threads, FLAGS.num_intra_threads)
else:
    PNet = FcnDetector(P_Net, model_path[0], FLAGS.num_inter_threads, FLAGS.num_intra_threads)
detectors[0] = PNet

# load rnet model
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1], FLAGS.num_inter_threads, FLAGS.num_intra_threads)
    detectors[1] = RNet

# load onet model
if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2], FLAGS.num_inter_threads, FLAGS.num_intra_threads)
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
gt_imdb = []
# gt_imdb.append("35_Basketball_Basketball_35_515.jpg")
# imdb_ = dict()"
# imdb_['image'] = im_path
# imdb_['label'] = 5
# path = "lala"
path = FLAGS.dl
# path = "prepare_data/WIDER_train/images/0--Parade"
for item in os.listdir(path):
    gt_imdb.append(os.path.join(path, item))
test_data = TestLoader(gt_imdb)

start = time.time()
all_boxes, landmarks = mtcnn_detector.detect_face(test_data)
end = time.time()
count = 0

accuracy = 0
for imagepah in gt_imdb:
    for bbox in all_boxes[count]:
        accuracy = accuracy + bbox[4]
    count = count + 1
accuracy = accuracy / count
print("Accuracy: %.2f" % (accuracy))

latency = (end - start) / count * 1000
tpt = count / (end - start)
print("Total images: %d" % count)
print("Latency is: %.2f, Throughput is: %.2f" % (latency, tpt))

"""
count = 0
for imagepath in gt_imdb:
    print imagepath
    image = cv2.imread(imagepath)
    for bbox in all_boxes[count]:
        cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))

    for landmark in landmarks[count]:
        for i in range(len(landmark)/2):
            cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))

    count = count + 1
    cv2.imwrite("result_landmark/%d.png" %(count),image)
    #cv2.imshow("lala",image)
    #cv2.waitKey(0)    
"""
'''
for data in test_data:
    print type(data)
    for bbox in all_boxes[0]:
        print bbox
        print (int(bbox[0]),int(bbox[1]))
        cv2.rectangle(data, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
    #print data
    cv2.imshow("lala",data)
    cv2.waitKey(0)
'''
