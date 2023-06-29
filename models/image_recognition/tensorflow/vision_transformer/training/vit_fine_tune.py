#
# -*- coding: utf-8 -*-
#
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
#
#

import time
import numpy as np

import tensorflow as tf
from transformers import TFViTForImageClassification
from argparse import ArgumentParser

print("TF version:", tf.__version__)

def get_arg_parser():
    arg_parser = ArgumentParser(description='Parse args')
    arg_parser.add_argument(
        '-d', '--data-location',
        help='Specify the location of the training data',
        dest='data_location', required=True
    )
    arg_parser.add_argument(
        '-p', '--precision',
        help='Specify the datatyp precision : fp32 / bfloat16 / fp16',
        dest='precision', default='fp32'
    )    
    arg_parser.add_argument(
        '-s', '--steps',
        help='Specify the number of steps for training',
        type=int, dest='steps',
        default=30000
    )
    arg_parser.add_argument(
        '-b', '--batch-size',
        help='Batch size for training',
        type=int, dest='batch_size',
        default=512
    )
    arg_parser.add_argument(
        '-i', '--init-checkpoint',
        help='Specify the location of the output directory for logs and saved model',
        dest='init_checkpoint', required=True
    )
    
    arg_parser.add_argument(
        '-o', '--model-dir',
        help='Specify the location of the output directory for logs and saved model',
        dest='model_dir', default='/tmp/vit-finetuned-model'
    )
    return arg_parser

parser = get_arg_parser()
args = parser.parse_args()


model_name = "vit_b32_imagenet21k"

model_handle_map = {
  "vit_b32_imagenet21k": args.init_checkpoint
}

model_image_size_map = {
  "vit_b32_imagenet21k": 224,
}

model_handle = model_handle_map.get(model_name)
pixels = model_image_size_map.get(model_name, 224)

print(f"Selected model: {model_name} : {model_handle}")

IMAGE_SIZE = (pixels, pixels)
print(f"Input size {IMAGE_SIZE}")

BATCH_SIZE = args.batch_size

############ Using ImageNet tf-records############
### reference : https://keras.io/examples/keras_recipes/tfrecord/
#######

TRAINING_FILENAMES = tf.io.gfile.glob(args.data_location + "/train-*")
VALID_FILENAMES = tf.io.gfile.glob(args.data_location + "/validation-*")

print("Train TFRecord Files:", len(TRAINING_FILENAMES))
print("Validation TFRecord Files:", len(VALID_FILENAMES))

#decode imagenet data
#The images have to be converted to tensors so that it will be a valid input in our model. As images utilize an RBG scale, we specify 3 channels.

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    return image

def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.
  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):
    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>
  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
  img = decode_image(features['image/encoded'])
  label = tf.cast(features['image/class/label'], dtype=tf.int32)
  label = tf.subtract(label, 1)

  return img, label


def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files

    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(_parse_example_proto, num_parallel_calls=tf.data.AUTOTUNE)
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset

def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_ds_total = get_dataset(TRAINING_FILENAMES)
NUM_TRAIN_SAMPLES = 1281167
num_train_split = int(0.99 * NUM_TRAIN_SAMPLES)
num_val_split = int(0.01 * NUM_TRAIN_SAMPLES)
test_ds = get_dataset(VALID_FILENAMES)

#Minval split 99%train - 1%validation
train_ds = train_ds_total.take(num_train_split)
val_ds = train_ds_total.skip(num_train_split).take(num_val_split)

def preprocess_data(image, train:bool):
    # For training, do a random crop and horizontal flip.
    if train:
        channels = image.shape[-1]
        begin, size, _ = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                tf.zeros([0, 0, 4], tf.float32),
                area_range=(0.05, 1.0),
                min_object_covered=0,
                use_image_if_no_bounding_boxes=True,
        )
        image = tf.slice(image, begin, size)
        image.set_shape([None, None, channels])
        image = tf.image.resize(image, IMAGE_SIZE)
        if tf.random.uniform(shape=[]) > 0.5:
            image = tf.image.flip_left_right(image)
    else:
        image = tf.image.resize(image, IMAGE_SIZE)
    image = (image - 127.5) / 127.5
    image =  tf.transpose(image, perm=[2,0,1])
    return image

train_ds = train_ds.map(lambda images, labels:
                        (preprocess_data(images, True), labels)).batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(lambda images, labels:
                    (preprocess_data(images, False), labels)).batch(BATCH_SIZE, drop_remainder=True)
test_ds = test_ds.map(lambda images, labels:
                    (preprocess_data(images, False), labels)).batch(BATCH_SIZE, drop_remainder=True)

do_fine_tuning = True

# Load the pre-trained Vision Transformer model
model = TFViTForImageClassification.from_pretrained(model_handle)

if not do_fine_tuning:
    for layer in model.layers[:-1]:
        layer.trainable = False

model.summary()

#Cosine Learning rate scheduler

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )
    def get_config(self):
        config = {'d_model': self.d_model, 'warmup_steps': self.warmup_steps}
        return config

TOTAL_STEPS = args.steps
WARMUP_STEPS = 50
INIT_LR = 0.03
WAMRUP_LR = 0.01
EPOCHS = int(TOTAL_STEPS / (num_train_split/BATCH_SIZE))

scheduled_lrs = WarmUpCosine(
    learning_rate_base=INIT_LR,
    total_steps=TOTAL_STEPS,
    warmup_learning_rate=WAMRUP_LR,
    warmup_steps=WARMUP_STEPS,
)

if args.precision == 'bfloat16':
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16':True})
elif args.precision == 'fp16':
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision':True})

optimizer = tf.keras.optimizers.SGD(scheduled_lrs, momentum=0.9, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.throughput = []

    def on_batch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_batch_end(self, batch, logs={}):
        total_time = time.time() - self.epoch_time_start
        self.times.append(total_time)
        self.throughput.append(BATCH_SIZE/total_time)

time_callback = TimeHistory()

hist = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds, callbacks=[time_callback]).history


# Evaluate the model
test_loss, test_acc = model.evaluate(test_ds)
print('Imagenet Validation set : Test accuracy:', test_acc)


avg_throughput = sum(time_callback.throughput)/len(time_callback.throughput)
print("Avg Throughput: " + str(avg_throughput) + " imgs/sec")
export_path = args.model_dir
model.save(export_path)
