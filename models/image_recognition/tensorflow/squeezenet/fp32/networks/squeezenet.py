from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d
from tensorflow.contrib.layers import batch_norm, l2_regularizer
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope


@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with arg_scope([conv2d, max_pool2d]):
            net = _squeeze(inputs, squeeze_depth)
            net = _expand(net, expand_depth)
        return net


def _squeeze(inputs, num_outputs):
    # v1.0 & v1.1
    return conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def _expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat([e1x1, e3x3], 1)


class Squeezenet(object):
    """v1.1 squeezenet architecture for 227x227 images."""
    name = 'squeezenet_v11'

    def __init__(self, args):
        self._num_classes = args.num_classes
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False

    def build(self, x, is_training):
        self._is_built = True
        with tf.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(is_training,self._weight_decay,
                                      self._batch_norm_decay)):
                return self._squeezenet(x, self._num_classes, is_training)

    def __conv2d(self, input_tensor, num_outputs, kernel_size, stride=1,
        scope=None, is_training=True, activation=tf.nn.relu):
        return conv2d(input_tensor, num_outputs, kernel_size, stride=stride,
                      scope=scope, data_format="NCHW",
                      activation_fn=activation, weights_regularizer=
                      l2_regularizer(self._weight_decay),normalizer_fn=
                      batch_norm, normalizer_params={
                        'is_training': is_training,
                        'fused': True,
                        'decay': self._batch_norm_decay})
        
    def __squeeze(self, input_tensor, squeeze_depth):
        return self.__conv2d(input_tensor, squeeze_depth, [1, 1],
                             scope='squeeze')  # , activation=None)

    def __expand(self, input_tensor, expand_depth):
        with tf.variable_scope('expand'):
            expand_1by1 = self.__conv2d(input_tensor, expand_depth,
                                        [1, 1], scope='1x1')
            expand_3by3 = self.__conv2d(input_tensor, expand_depth,
                                        [3, 3], scope='3x3')
        return tf.concat([expand_1by1, expand_3by3], 1)

    def _fire_module(self, input_tensor, squeeze_depth, expand_depth,
                     reuse=None, scope=None):
        with tf.variable_scope(scope, 'fire', [input_tensor], reuse=reuse):
            squeeze_tensor = self.__squeeze(input_tensor, squeeze_depth)
            expand_tensor = self.__expand(squeeze_tensor, expand_depth)
        return expand_tensor

    def _squeezenet(self,images, num_classes,is_training):
        net = self.__conv2d(images, 64, [3, 3], stride=2, scope='conv1')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
        net = self._fire_module(net, 16, 64, scope='fire2')
        net = self._fire_module(net, 16, 64, scope='fire3')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool3')
        net = self._fire_module(net, 32, 128, scope='fire4')
        net = self._fire_module(net, 32, 128, scope='fire5')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool5')
        net = self._fire_module(net, 48, 192, scope='fire6')
        net = self._fire_module(net, 48, 192, scope='fire7')
        net = self._fire_module(net, 64, 256, scope='fire8')
        net = self._fire_module(net, 64, 256, scope='fire9')
        net = tf.layers.dropout(net, rate=0.5, training=is_training,
                                name="drop9")
        net = self.__conv2d(net, num_classes, [1, 1], stride=1, scope='conv10')
        net = avg_pool2d(net, [13, 13], stride=1, scope='avgpool10')
        logits = tf.squeeze(net, [2], name='logits')
        return logits


class SqueezenetV1(object):
    """v1.0 squeezenet architecture for 224x224 images."""
    name = 'squeezenet_v10'

    def __init__(self, args):
        self._num_classes = args.num_classes
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False

    def build(self, x, is_training):
        self._is_built = True
        with tf.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(is_training,
                                      self._weight_decay,
                                      self._batch_norm_decay)):
                return self._squeezenet(x, self._num_classes,is_training)

    @staticmethod
    def _squeezenet(images, num_classes, is_training):
        net = conv2d(images, 96, [7, 7], stride=2, scope='conv1')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
        net = fire_module(net, 16, 64, scope='fire2')
        net = fire_module(net, 16, 64, scope='fire3')
        net = fire_module(net, 32, 128, scope='fire4')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool4')
        net = fire_module(net, 32, 128, scope='fire5')
        net = fire_module(net, 48, 192, scope='fire6')
        net = fire_module(net, 48, 192, scope='fire7')
        net = fire_module(net, 64, 256, scope='fire8')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
        net = fire_module(net, 64, 256, scope='fire9')
        net = tf.layers.dropout(net, rate=0.5,training=is_training,
                                name="drop9")
        net = conv2d(net, num_classes, [1, 1], stride=1, scope='conv10')
        net = avg_pool2d(net, [13, 13], stride=1, scope='avgpool10')
        logits = tf.squeeze(net, [2], name='logits')
        return logits


class Squeezenet_CIFAR(object):
    """Modified version of squeezenet for CIFAR images"""
    name = 'squeezenet_cifar'

    def __init__(self, args):
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False

    def build(self, x, is_training):
        self._is_built = True
        with tf.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(is_training,
                                      self._weight_decay,
                                      self._batch_norm_decay)):
                return self._squeezenet(x)

    @staticmethod
    def _squeezenet(images, num_classes=10):
        net = conv2d(images, 96, [2, 2], scope='conv1')
        net = max_pool2d(net, [2, 2], scope='maxpool1')
        net = fire_module(net, 16, 64, scope='fire2')
        net = fire_module(net, 16, 64, scope='fire3')
        net = fire_module(net, 32, 128, scope='fire4')
        net = max_pool2d(net, [2, 2], scope='maxpool4')
        net = fire_module(net, 32, 128, scope='fire5')
        net = fire_module(net, 48, 192, scope='fire6')
        net = fire_module(net, 48, 192, scope='fire7')
        net = fire_module(net, 64, 256, scope='fire8')
        net = max_pool2d(net, [2, 2], scope='maxpool8')
        net = fire_module(net, 64, 256, scope='fire9')
        net = avg_pool2d(net, [4, 4], scope='avgpool10')
        net = conv2d(net, num_classes, [1, 1],
                     activation_fn=None,
                     normalizer_fn=None,
                     scope='conv10')
        logits = tf.squeeze(net, [2], name='logits')
        return logits


def _arg_scope(is_training, weight_decay, bn_decay):
    with arg_scope([conv2d],
                   weights_regularizer=l2_regularizer(weight_decay),
                   normalizer_fn=batch_norm,
                   normalizer_params={'is_training': is_training,
                                      'fused': True,
                                      'decay': bn_decay}):
        with arg_scope([conv2d, avg_pool2d, max_pool2d, batch_norm],
                       data_format='NCHW') as sc:
                return sc

'''
Network in Network: https://arxiv.org/abs/1312.4400
See Section 3.2 for global average pooling
'''
