import tensorflow as tf
import argparse
import numpy as np
import time

def main(num_intra_threads,num_inter_threads, batch_size, warmup_steps, steps, input_graph):
    #Load Tensorflow pb graph
    infer_graph = tf.Graph()    
    with infer_graph.as_default():
        graph_def = tf.compat.v1.GraphDef() 
        with tf.compat.v1.gfile.FastGFile(input_graph, 'rb') as input_file:
            input_graph_content = input_file.read()
            graph_def.ParseFromString(input_graph_content)
        
        tf.import_graph_def(graph_def, name='')


    infer_config = tf.compat.v1.ConfigProto()
    infer_config.intra_op_parallelism_threads = num_intra_threads
    infer_config.inter_op_parallelism_threads = num_inter_threads
    infer_config.use_per_session_threads = 1


    input_tensor1 = infer_graph.get_tensor_by_name('v0/cg/mpool1/MaxPool:0')
    output_tensor = infer_graph.get_tensor_by_name('v0/cg/incept_v3_a0/concat:0')

    infer_sess = tf.compat.v1.Session(graph=infer_graph, config=infer_config)

    image_np1 = np.zeros((batch_size,35,35,192))

    iteration = 0
    total_time = 0
    
    while iteration < steps:
        iteration += 1

        start_time = time.time()
        infer_sess.run([output_tensor], feed_dict={input_tensor1: image_np1})
        time_consume = time.time()-start_time
        print('Iteration %d: %.6f sec' % (iteration, time_consume))
        
        if iteration > warmup_steps:
            total_time += time_consume

    time_average = total_time / (iteration - warmup_steps)
    
    print('--------------------------------------------------')
    print('Batch size is                : ', batch_size)
    print('Number of Intra Threads      : ', num_intra_threads)
    print('Number of Inter Threads      : ', num_inter_threads)
    if (batch_size == 1):
        print('Latency  (seconds)           : ', time_average)
    print('Throughput    (images/sec)   : ', (batch_size / time_average))
    print('--------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_graph", type=str, default=None, help="path of pb graph")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Enter width of graph to verify")
    parser.add_argument("--num_intra_threads", type=int, default=1, help="Enter number of intra threads")
    parser.add_argument("--num_inter_threads", type=int, default=1, help="Enter number of inter threads")
    parser.add_argument("--batch_size", type=int, default=1, help="Enter batch size")
    parser.add_argument("--steps", type=int, default=20, help="Enter number of steps(Iterations)")
    args = parser.parse_args()
    main(**args.__dict__)
