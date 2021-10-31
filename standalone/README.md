This folder consits creation of subgraphs of few patterns that can be used in standalone scripts.

The file "get_inception_graph_keras_api_6conv.py" helps to get a model with 6 parallel convolutions using keras API. The model gets saved when we run the script. Using the saved pb model we can run the benchark sctipt of inception from intelAI(launch_benchmarks.py) bydoing few changes in eval_image_classifer.py(specific to Inceptionv3) such as input nodes,output nodes and input.


The file "get_inception_graph_tf_api_sequential.py" helps to create a model of sequential pattern using TF API. The same script helps to run the benchmark.

Command to run:
python get_inception_graph_tf_api_sequential.py --intra_threads=32 --inter_threads=2 --batch_size=64

The file "graph_cut_user_level.py" helps to get the subgraph from the inceptionv3 graph at user level.
We need to specify the inputs and ouptus in the file to do the graph cut.

After doing the graphcut from the inceptionv3 model
We can run the benchmark for the model using "benchmark-module1.py"
Note:During the graphcut if the inputs and outputs are changed while saving the model then while running the benchmark we need to change the inputs and outputs as per the graph that is saved[Input to be passed, input and output nodes].

Command to run:
numactl --cpunodebind=0 --interleave=0 python benchmark-module1.py --input_graph {graph path} --num_intra_threads=32 --num_inter_threads=2 --batch_size=64
