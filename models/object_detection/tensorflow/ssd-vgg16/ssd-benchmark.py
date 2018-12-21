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
from google.protobuf import text_format
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline

import os
import numpy as np
import math
from preparedata import PrepareData
from nets.ssd import g_ssd_model
import tf_extended as tfe
import time
from postprocessingdata import g_post_processing_data
import argparse

os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["OMP_PROC_BIND"] = "true"

# For python3
def flatten(lst):
    """Flattens a list of lists"""
    return [subelem for elem in lst 
                    for subelem in elem]

class EvaluateModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)

        self.batch_size = 1
        self.labels_offset = 0
        self.use_dummy_data = True
        self.eval_image_size = None
        self.data_format = 'NHWC'
        self.num_preprocessing_threads = 4
        self.num_interop_threads = 2
        self.num_intraop_threads = 28
        self.checkpoint_path =  None
        self.accuracy_check = False
        self.image_width    = 300
        self.image_height   = 300

        return
    
    
    def load_inference_graph(self, model_file):
      graph = tf.Graph()
      graph_def = tf.GraphDef()

      import os
      file_ext = os.path.splitext(model_file)[1]

      with open(model_file, "rb") as f:
        if file_ext == '.pbtxt':
          text_format.Merge(f.read(), graph_def)
        else:
          graph_def.ParseFromString(f.read())
      with graph.as_default():
        tf.import_graph_def(graph_def)
      return graph

    def load_accuracy_graph(self):
      #Create the graph to check accuracy
      accuracy_graph = tf.Graph()
      
      with accuracy_graph.as_default():
        predict0 = tf.placeholder(tf.float32, [self.batch_size,38, 38, 4, 21], "prediction_0") 
        predict1 = tf.placeholder(tf.float32, [self.batch_size,19, 19, 6, 21], "prediction_1") 
        predict2 = tf.placeholder(tf.float32, [self.batch_size,10, 10, 6, 21], "prediction_2") 
        predict3 = tf.placeholder(tf.float32, [self.batch_size,5, 5, 6, 21],   "prediction_3") 
        predict4 = tf.placeholder(tf.float32, [self.batch_size,3, 3, 4, 21],   "prediction_4") 
        predict5 = tf.placeholder(tf.float32, [self.batch_size,1, 1, 4, 21],   "prediction_5") 
        predictions   = [predict0, predict1, predict2, predict3, predict4, predict5]

        localisation0 = tf.placeholder(tf.float32, [self.batch_size,38, 38, 4, 4], "localisation_0") 
        localisation1 = tf.placeholder(tf.float32, [self.batch_size,19, 19, 6, 4], "localisation_1") 
        localisation2 = tf.placeholder(tf.float32, [self.batch_size,10, 10, 6, 4], "localisation_2") 
        localisation3 = tf.placeholder(tf.float32, [self.batch_size,5, 5, 6, 4],   "localisation_3") 
        localisation4 = tf.placeholder(tf.float32, [self.batch_size,3, 3, 4, 4],   "localisation_4") 
        localisation5 = tf.placeholder(tf.float32, [self.batch_size,1, 1, 4, 4],   "localisation_5") 
        localisations = [localisation0, localisation1, localisation2, 
                       localisation3, localisation4, localisation5]

        glabels       = tf.placeholder(tf.int64,   [self.batch_size,None],           "glabels")
        gbboxes       = tf.placeholder(tf.float32, [self.batch_size,None, 4],        "gbboxes")
        gdifficults   = tf.placeholder(tf.float32, [self.batch_size,None],           "gdifficults")
        names_to_updates, mAP_reports = g_post_processing_data.get_mAP_tf_accumulative(
                      predictions, localisations, glabels, gbboxes, gdifficults)

      #with open("./ssd-accuracy.pbtxt", "w") as f:
      #  f.write(str(accuracy_graph.as_graph_def()))
      #  f.close()
      place_holders = [predict0, predict1, predict2, predict3, predict4, predict5,
                       localisation0, localisation1, localisation2, 
                       localisation3, localisation4, localisation5,
                       glabels, gbboxes, gdifficults]

      return accuracy_graph, place_holders, flatten(list(names_to_updates.values())), mAP_reports

    def eval(self):
      
      inference_graph = self.load_inference_graph(self.graph_file)
      input_name = "import/input"
      output_names = [#predictions
                      "import/ssd_300_vgg/softmax/Reshape_1",
                      "import/ssd_300_vgg/softmax_1/Reshape_1",
                      "import/ssd_300_vgg/softmax_2/Reshape_1",
                      "import/ssd_300_vgg/softmax_3/Reshape_1",
                      "import/ssd_300_vgg/softmax_4/Reshape_1",
                      "import/ssd_300_vgg/softmax_5/Reshape_1",
                      #localisations
                      "import/ssd_300_vgg/block4_box/Reshape",
                      "import/ssd_300_vgg/block7_box/Reshape",
                      "import/ssd_300_vgg/block8_box/Reshape",
                      "import/ssd_300_vgg/block9_box/Reshape",
                      "import/ssd_300_vgg/block10_box/Reshape",
                      "import/ssd_300_vgg/block11_box/Reshape"]
      input_operation = inference_graph.get_operation_by_name(input_name);
      output_operations = []
      for name in output_names:
        output_operations.append(inference_graph.get_operation_by_name(name).outputs[0])
      
      accuracy_graph, accuracy_place_holders, accu_eval_ops, accu_report_ops = self.load_accuracy_graph()

      config = tf.ConfigProto()
      config.intra_op_parallelism_threads = self.num_intraop_threads
      config.inter_op_parallelism_threads = self.num_interop_threads

      run_options = None
      run_metadata = None

      if self.use_dummy_data == True:
        with tf.Session(config=config) as sess:
          dummy_image = sess.run(tf.truncated_normal(
              [self.batch_size, self.image_height, self.image_width, 3],
              dtype=tf.float32,
              stddev=10,
              name='synthetic_images'))

        with tf.Session(graph=inference_graph, config=config) as sess1:
          total_time = 0.0
          if (self.batch_size == 1):
            num_warmup_batches = 1000
          else:
            num_warmup_batches = 10
          print(num_warmup_batches)
          for i in range(num_warmup_batches):  
            if ( (i+1) % 10 == 0):
              print("Run warmup batch [%d/%d]" % (i+1, num_warmup_batches))
            sess1.run(output_operations, {input_operation.outputs[0] : dummy_image},
                      options=run_options, run_metadata=run_metadata)

          print("benchmark run")
          if (self.batch_size == 1):
            num_batches = 1000
          else:
            num_batches = 100
          timeline_step = -1
          for i in range(num_batches):
            if (i == timeline_step):
              run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
              run_metadata = tf.RunMetadata()
            else:
              run_options = None
              run_metadata = None    

            start_time = time.time()
            sess1.run(output_operations, {input_operation.outputs[0] : dummy_image},
                      options=run_options, run_metadata=run_metadata)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            if (i == timeline_step):
              trace = timeline.Timeline(step_stats=run_metadata.step_stats)
              dir = 'timeline/'
              if not os.path.exists(dir):
                os.makedirs(dir)
              with open(dir + time.strftime("%Y%m%d-%H%M%S") + '.json', 'w') as file:
                file.write(trace.generate_chrome_trace_format(show_memory=False))
            if ( (i+1) % 10 == 0):
              print("Run benchmark batch [%d/%d]: %f images/sec" % (i+1, num_batches,  
                (i+1) * self.batch_size / total_time))
        return

      with tf.Graph().as_default():
        gimage, _, glabels,gbboxes,gdifficults, _, _, _ = self.get_voc_2007_test_data(self.data_location)
        num_batches = int(math.floor(self.dataset.num_samples / float(self.batch_size)))

        with tf.Session(config=config) as sess:
          init = tf.global_variables_initializer()
          sess.run(init)

          with slim.queues.QueueRunners(sess):
            with  tf.Session(graph=accuracy_graph, config=config) as sess_accu:
              sess_accu.run(tf.local_variables_initializer())

              total_time = 0.0
              timeline_step = -1
              for i in range(num_batches):
                o_image, o_labels, o_bboxes, o_difficults = sess.run([gimage, glabels, 
                      gbboxes, gdifficults], options=run_options, run_metadata=run_metadata)
                
                if (i == timeline_step):
                  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                  run_metadata = tf.RunMetadata()
                else:
                  run_options = None
                  run_metadata = None 
  
                start_time = time.time()
                with tf.Session(graph=inference_graph, config=config) as sess1:
                  output = sess1.run(output_operations, {input_operation.outputs[0] : o_image},
                          options=run_options, run_metadata=run_metadata)
                elapsed_time = time.time() - start_time
                total_time += elapsed_time

                if (i == timeline_step):
                  trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                  dir = 'timeline/'
                  if not os.path.exists(dir):
                    os.makedirs(dir)
                  with open(dir + time.strftime("%Y%m%d-%H%M%S") + '.json', 'w') as file:
                    file.write(trace.generate_chrome_trace_format(show_memory=False))
 
                print("Run batch [%d/%d]: %f images/sec" % (i+1, num_batches,
                    (i+1) * self.batch_size / total_time))

                if self.accuracy_check == True:
                  sess_accu.run(accu_eval_ops, 
                     {accuracy_place_holders[0]  : output[0],
                      accuracy_place_holders[1]  : output[1],
                      accuracy_place_holders[2]  : output[2],
                      accuracy_place_holders[3]  : output[3],
                      accuracy_place_holders[4]  : output[4],
                      accuracy_place_holders[5]  : output[5],
                      accuracy_place_holders[6]  : output[6],
                      accuracy_place_holders[7]  : output[7],
                      accuracy_place_holders[8]  : output[8],
                      accuracy_place_holders[9]  : output[9],
                      accuracy_place_holders[10] : output[10],
                      accuracy_place_holders[11] : output[11],
                      accuracy_place_holders[12] : o_labels, 
                      accuracy_place_holders[13] : o_bboxes, 
                      accuracy_place_holders[14] : o_difficults
                     },
                     options=run_options, run_metadata=run_metadata)

              #end of for loop
              if self.accuracy_check == True:
                accuracy_report = sess_accu.run(accu_report_ops, 
                     options=run_options, run_metadata=run_metadata)
                print( "mAP_VOC12 = %f, mAP_VOC07 = %f" % (accuracy_report[1], accuracy_report[0]))
      return

    def parse_param(self):
        arg_parser = argparse.ArgumentParser()

        arg_parser.add_argument('-a', '--accuracy_check',help='Run accuracy check',  action='store_true')
        arg_parser.add_argument('-b', '--batch_size',    type=int, help='batch size',          default='224')
        arg_parser.add_argument('-u', '--use_voc_data',  help='Use doc data',        action='store_true')
        arg_parser.add_argument('-g', '--graph',         help='the input graph',     default="final_intel_qmodel_ssd.pb")
        #arg_parser.add_argument('-g', '--graph',         help='the input graph',     default="freezed_ssd.pbtxt")
        arg_parser.add_argument('-r', "--num_intra_threads", type=int, help="Specify the number of threads within the layer", 
                          dest="intra_op", default=None)
        arg_parser.add_argument('-e', "--num_inter_threads", type=int, help='Specify the number threads between layers', 
                          dest="inter_op", default=None)
        arg_parser.add_argument('-d', "--data_location",
                                help='Specify the location of the data. '
                                'If this parameter is not specified, '
                                'the benchmark will use random/dummy data.',
                                 dest="data_location", default=None)

        args = arg_parser.parse_args()

        self.accuracy_check = args.accuracy_check
        self.batch_size     = args.batch_size
        self.use_dummy_data = not args.use_voc_data
        self.graph_file     = args.graph
        self.num_intraop_threads = args.intra_op
        self.num_interop_threads = args.inter_op     
        self.data_location = args.data_location

        return
    
if __name__ == "__main__":   
    obj= EvaluateModel()
    obj.parse_param()
    obj.eval()
