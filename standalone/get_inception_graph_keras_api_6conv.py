"TF-2-with Keras model"
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


# Model Function
def conv_net_save():
    inputs = tf.keras.Input(shape=(299,299,3),name='input')
    conv1a = tf.keras.layers.Conv2D(320, 1, activation="relu", padding = 'same')(inputs)
    conv2a = tf.keras.layers.Conv2D(384, (1,3), activation="relu", padding = 'same')(inputs)
    conv3a = tf.keras.layers.Conv2D(384, (3,1), activation="relu", padding = 'same')(inputs)
    conv4a = tf.keras.layers.Conv2D(384, (1,3), activation="relu", padding = 'same')(inputs)
    conv5a = tf.keras.layers.Conv2D(384, (3,1), activation="relu", padding = 'same')(inputs)
    conv6a = tf.keras.layers.Conv2D(192, 1, activation="relu", padding = 'same')(inputs)
    con2 = tf.keras.layers.concatenate([conv1a,conv2a,conv3a,conv4a, conv5a, conv6a],3,name='output2')
                 
    #model = tf.keras.Model(inputs=inputs, outputs=con1, name="model")

    model = tf.keras.Model(inputs=inputs, outputs=con2, name="model")
    model.summary()

    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir='',
                  name="graph-6conv.pb",
                  as_text=False)

    print("Model saved")
    #The Model that is saved can be used for inception Inference

def main(unused_argv):
  del unused_argv

  print('Tensorflow version: ' + str(tf.__version__))
  conv_net_save()    


if __name__ == "__main__":
  tf.compat.v1.app.run()
