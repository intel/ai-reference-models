import tensorflow as tf
import time
import numpy as np
import os

from tensorflow.python.client import timeline

tf.compat.v1.disable_v2_behavior()

tf.compat.v1.flags.DEFINE_integer("intra_threads", 1, "For best performance, set it to be the number of physical cores per socket")
tf.compat.v1.flags.DEFINE_integer("inter_threads", 1, "For best performance, set to be the number of sockets.")
tf.compat.v1.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.compat.v1.flags.DEFINE_integer("iteration", 30, "Numebr of iterations")

FLAGS = tf.compat.v1.flags.FLAGS

# 2D Convolutional Function
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')
#    x = tf.nn.bias_add(x, b)
    return x

# Define Weights and Biases
weights = {
    # Convolution Layers
    'c1': tf.compat.v1.get_variable('W1', shape=(3,3,3,32)), 
    'c2': tf.compat.v1.get_variable('W2', shape=(3,3,32,32)),
    'c3': tf.compat.v1.get_variable('W3', shape=(3,3,32,64)),
    'c4': tf.compat.v1.get_variable('W4', shape=(1,1,64,80)),
    'c5': tf.compat.v1.get_variable('W5', shape=(3,3,80,192))
}

biases = {
    # Convolution Layers
    'c1': tf.compat.v1.get_variable('B1', shape=(32)),
    'c2': tf.compat.v1.get_variable('B2', shape=(32)),
    'c3': tf.compat.v1.get_variable('B3', shape=(64)),
    'c4': tf.compat.v1.get_variable('B4', shape=(80)),
    'c5': tf.compat.v1.get_variable('B5', shape=(192))
}

# Model Function
def conv_net(weights, biases):
    # Convolution layers
    inputs = tf.compat.v1.placeholder(dtype=tf.float32,shape=(None,299,299,3),name='input')
    conv1 = tf.nn.conv2d(inputs, weights['c1'], strides=[1,2,2,1], padding='VALID') 
    bn1 = tf.nn.batch_normalization(conv1, 32, 32, 32, 32, 0.0010000000474974513)
    relu1 = tf.nn.relu(bn1)
    conv2 = conv1 = tf.nn.conv2d(relu1, weights['c2'], strides=[1,1,1,1], padding='VALID')
    bn2 = tf.nn.batch_normalization(conv2, 32, 32, 32, 32, 0.0010000000474974513)
    relu2 = tf.nn.relu(bn2)
    conv3 = conv1 = tf.nn.conv2d(relu2, weights['c3'], strides=[1,1,1,1], padding='SAME')
    bn3 = tf.nn.batch_normalization(conv3, 64, 64, 64, 64, 0.0010000000474974513)
    relu3 = tf.nn.relu(bn3)
    conv4 = conv1 = tf.nn.conv2d(relu3, weights['c4'], strides=[1,1,1,1], padding='VALID')
    bn4 = tf.nn.batch_normalization(conv4, 80, 80, 80, 80, 0.0010000000474974513)
    relu4 = tf.nn.relu(bn4)
    conv5 = conv1 = tf.nn.conv2d(relu4, weights['c5'], strides=[1,1,1,1], padding='VALID')
    bn5 = tf.nn.batch_normalization(conv5, 192, 192, 192, 192, 0.0010000000474974513)
    relu5 = tf.nn.relu(bn5)
#    out = tf.concat([conv1,conv2,conv3,conv4],3,name='output')


def main(unused_argv):
  del unused_argv

  print('Tensorflow version: ' + str(tf.__version__))

  config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=FLAGS.intra_threads, \
              inter_op_parallelism_threads=FLAGS.inter_threads)

  conv_net(weights, biases)

  tf.compat.v1.disable_eager_execution()
  g = tf.compat.v1.get_default_graph()
  tf.io.write_graph(g.as_graph_def(), '', 'graph.pb', as_text=False)
  
  #all_tensors = [tensor for op in g.get_operations() for tensor in op.values()]
  #print(all_tensors)

  sess = tf.compat.v1.Session(config=config)
  sess.run(tf.compat.v1.global_variables_initializer())
  sess.run(tf.compat.v1.local_variables_initializer())

  image_np = np.zeros((FLAGS.batch_size,299,299,3)) 
  input_tensor = g.get_tensor_by_name('input:0')
  output_tensor = g.get_tensor_by_name('Relu_5:0')
  
  iteration=0
  total_time=0
#  options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)  
#  run_metadata = tf.compat.v1.RunMetadata()
  while iteration < FLAGS.iteration:
    iteration += 1

    start_time = time.time()
    sess.run([output_tensor], feed_dict={input_tensor: image_np})
    time_consume = time.time()-start_time
    print('Iteration %d: %.6f sec' % (iteration, time_consume))
    if iteration > 10:
        total_time += time_consume

  time_average = total_time / (iteration - 10)
#  trace = timeline.Timeline(run_metadata.step_stats)
  if (FLAGS.batch_size == 1):
    print('Latency  (seconds)           : ', time_average)
  print('Throughput    (images/sec)   : ', (FLAGS.batch_size / time_average))
  
#  if (FLAGS.batch_size == 1):
#      filename = 'inception-v3_Intra-'+str(FLAGS.intra_threads)+'_Inter-'+str(FLAGS.inter_threads)+'_BS-'+str(FLAGS.batch_size)+'_Throughput-'+str('%.3f'%(FLAGS.batch_size / time_average))+'_Latency-'+str('%.3f'%(time_average))+'.json'
#  else:
#      filename = 'inception-v3_Intra-'+str(FLAGS.intra_threads)+'_Inter-'+str(FLAGS.inter_threads)+'_BS-'+str(FLAGS.batch_size)+'_Throughput-'+str('%.3f'%(FLAGS.batch_size / time_average))+'.json'

#  with open(filename, 'w') as trace_file:
      # writing timeline object
#      trace_file.write(trace.generate_chrome_trace_format(show_memory=True))




if __name__ == "__main__":
  tf.compat.v1.app.run()
