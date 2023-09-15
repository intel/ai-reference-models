# Copyright (c) 2023 Intel Corporation
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
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

import numpy as np
import argparse
import time
import tensorflow as tf
import tensorflow.keras.applications as tka
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

@tf.function
def tf_function_model_predict(model, x_input):
    return model(x_input, training = False)

def main(args):
    from tensorflow.keras import  mixed_precision                                                                                                         
    policy = mixed_precision.Policy('mixed_float16')
    #policy = mixed_precision.Policy('mixed_bfloat16')
    mixed_precision.set_global_policy(policy)

    img_size = None
    if args.model == "EfficientNetB0":
        img_size = (224, 224)
    elif args.model == "EfficientNetB3":
        img_size = (300, 300)
    elif args.model == "EfficientNetB4":
        img_size = (380, 380)
    else:
        assert (False and "error model name")

    print("load data ......")    
    img_path = args.image_file
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)#.astype(np.float32)

    rep = np.array([args.batch_size,], dtype = np.int32)
    x = tf.repeat(x, rep, axis = 0)

    print("input shape", x.shape) 
   
    model = getattr(tka, args.model)(weights='imagenet')
    model.trainable = False
    print("Creating model finished.")

    total_iter = 50
    warmup_iter = 20
    total_time = 0
    total_count = 0
    preds = None
    for step in range(total_iter):
        start = time.time()
        #preds = model.predict(x)
        #preds = model(x, training=False)
        preds = tf_function_model_predict(model, x)
        end = time.time()
        if step >= warmup_iter:
            total_time += (end - start)
            total_count += 1
    """
    # resnet50 result
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)
    """
    print("Batchsize is", args.batch_size)
    avg_time = total_time / total_count 
    print("Avg time:", avg_time, "s.")
    print("Throughput:", args.batch_size/avg_time, "img/s.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", required=True, help="model name")
    #parser.add_argument("-d", "--dtype", dest="dtype", required=True, help="data type")
    #parser.add_argument("-l", "--layout", dest="layout", required=True, help="data layout")
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, required=True, help="batch size")
    parser.add_argument("-i", "--image_file", dest="image_file", required=True, help="input ImageNet image")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
