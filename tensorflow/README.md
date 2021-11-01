# graph cutting at tf level
update optimize_for_inference_lib file present in the below path
/tensorflow/tensorflow/python/tools/optimize_for_inference_lib.py

1) sample execution statement :
    cuts = [
    [['input'],['v0/cg/mpool1/MaxPool']],
    [['v0/cg/mpool1/MaxPool'],['v0/cg/incept_v3_a0/concat']],
    [['v0/cg/incept_v3_a0/concat'],['predict']],
    ]
    output_graph = optimize_for_inference(graph_def, cuts=cuts, placeholder_type_enum=dtypes.float32.as_datatype_enum, toco_compatible=False)

will return a list of graphs cut in specified node locations mentioned in cuts
