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
#import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import numpy as np
import math
from preparedata import PrepareData
from nets.ssd import g_ssd_model
import tf_extended as tfe
import time
from tensorflow.python.ops import math_ops


class PostProcessingData(object):
    def __init__(self):
       
        
        
        return
    
    def __compute_AP(self,c_scores,c_tp,c_fp,c_num_gbboxes):
        aps_voc07 = {}
        aps_voc12 = {}
        for c in c_scores.keys():
            num_gbboxes = c_num_gbboxes[c]
            tp = c_tp[c]
            fp = c_fp[c]
            scores = c_scores[c]
        
            #reshape data
            num_gbboxes = math_ops.to_int64(num_gbboxes)
            scores = math_ops.to_float(scores)
            stype = tf.bool
            tp = tf.cast(tp, stype)
            fp = tf.cast(fp, stype)
            # Reshape TP and FP tensors and clean away 0 class values.(difficult bboxes)
            scores = tf.reshape(scores, [-1])
            tp = tf.reshape(tp, [-1])
            fp = tf.reshape(fp, [-1])
            
            # Remove TP and FP both false.
            mask = tf.logical_or(tp, fp)
    
            rm_threshold = 1e-4
            mask = tf.logical_and(mask, tf.greater(scores, rm_threshold))
            scores = tf.boolean_mask(scores, mask)
            tp = tf.boolean_mask(tp, mask)
            fp = tf.boolean_mask(fp, mask)
            
            num_gbboxes = tf.reduce_sum(num_gbboxes)
            num_detections = tf.size(scores, out_type=tf.int32)
            
            # Precison and recall values.
            prec, rec = tfe.precision_recall(num_gbboxes, num_detections, tp, fp, scores)
            
            v = tfe.average_precision_voc07(prec, rec)
            aps_voc07[c] = v
            
            # Average precision VOC12.
            v = tfe.average_precision_voc12(prec, rec)

            aps_voc12[c] = v
        return aps_voc07, aps_voc12
    
    def get_mAP_tf_current_batch(self,predictions, localisations,glabels, gbboxes,gdifficults):
        # Performing post-processing on CPU: loop-intensive, usually more efficient.
        with tf.device('/device:CPU:0'):
        
            # Detected objects from SSD output.
            localisations = g_ssd_model.decode_bboxes_all_ayers_tf(localisations)
            # Select via thresholding and also top_k bboxes from predictions
            # Apply NMS algorithm.
            rscores, rbboxes = g_ssd_model.detected_bboxes(predictions, localisations)
            
            # Compute TP and FP statistics.
            c_num_gbboxes, c_tp, c_fp, c_scores = \
                tfe.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                          glabels, gbboxes, gdifficults)
            
            
            aps_voc07, aps_voc12 = self.__compute_AP(c_scores, c_tp, c_fp, c_num_gbboxes)
            # Mean average precision VOC07.
#             summary_name = 'AP_VOC07/mAP'
            mAP_07_op = tf.add_n(list(aps_voc07.values())) / len(aps_voc07)
#             op = tf.summary.scalar(summary_name, mAP, collections=[])
#             print_mAP_07_op = tf.Print(mAP_07, [mAP_07], summary_name)
#             tf.summary.scalar(summary_name, print_mAP_07_op)
#             tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    
            # Mean average precision VOC12.
#             summary_name = 'AP_VOC12/mAP'
            mAP_12_op = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
#             op = tf.summary.scalar(summary_name, mAP, collections=[])
#             print_mAP_12_op = tf.Print(mAP, [mAP], summary_name)
#             tf.summary.scalar(summary_name, print_mAP_12_op)


            
        return mAP_07_op, mAP_12_op
    
    def get_mAP_tf_accumulative(self,predictions, localisations,glabels, gbboxes,gdifficults):
        # Performing post-processing on CPU: loop-intensive, usually more efficient.
        with tf.device('/device:CPU:0'):
        
            # Detected objects from SSD output.
            localisations = g_ssd_model.decode_bboxes_all_ayers_tf(localisations)
            
            rscores, rbboxes = g_ssd_model.detected_bboxes(predictions, localisations)
            
            # Compute TP and FP statistics.
            num_gbboxes, tp, fp, rscores = \
                tfe.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                          glabels, gbboxes, gdifficults)
        dict_metrics = {}
        with tf.device('/device:CPU:0'):
    
            # FP and TP metrics.
            tp_fp_metric = tfe.streaming_tp_fp_arrays(num_gbboxes, tp, fp, rscores)
            for c in tp_fp_metric[0].keys():
                dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c],
                                                    tp_fp_metric[1][c])
                
            # Add to summaries precision/recall values.
            aps_voc07 = {}
            aps_voc12 = {}
            for c in tp_fp_metric[0].keys():
                # Precison and recall values.
                prec, rec = tfe.precision_recall(*tp_fp_metric[0][c])
    
                # Average precision VOC07.
                v = tfe.average_precision_voc07(prec, rec)
                summary_name = 'AP_VOC07/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                # op = tf.Print(op, [v], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_voc07[c] = v
    
                # Average precision VOC12.
                v = tfe.average_precision_voc12(prec, rec)
                summary_name = 'AP_VOC12/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                # op = tf.Print(op, [v], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_voc12[c] = v
    
            # Mean average precision VOC07.
            mAP_report = []
            summary_name = 'AP_VOC07/mAP_accumulative'
            mAP = tf.add_n(list(aps_voc07.values())) / len(aps_voc07)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            mAP_report.append(mAP)
            
    
            # Mean average precision VOC12.
            summary_name = 'AP_VOC12/mAP_accumulative'
            mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            mAP_report.append(mAP)
            
            # Split into values and updates ops.
            names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)
        return names_to_updates, mAP_report
    
    def run(self):
        
        
       
        
        
        return
    
    
g_post_processing_data = PostProcessingData()

if __name__ == "__main__":   
    obj= PostProcessingData()
    obj.run()
