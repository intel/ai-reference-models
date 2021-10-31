import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from tensorflow.python.tools import strip_unused_lib

def run():
    input_node_names=['v0/cg/mpool1/MaxPool']
    output_node_names=['v0/cg/incept_v3_a0/concat']
    infer_graph = tf.Graph()
    with infer_graph.as_default():
      graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v1.gfile.FastGFile('inceptionv3_fp32_pretrained_model.pb', 'rb') as input_file:
        input_graph_content = input_file.read()
        graph_def.ParseFromString(input_graph_content)
        cut_graph = strip_unused_lib.strip_unused(graph_def, input_node_names, output_node_names,
        placeholder_type_enum=tf.float32.as_datatype_enum)
        tf.io.write_graph(cut_graph,'./', name='poc.pb',as_text=False)
run()
