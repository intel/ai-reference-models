This folder helps to run the benchmark for Inceptionv3/bert/wide_deep.
The automation script chooses the possible inter and intra threads based on the machine configuration.

To run the benchmark bash script we can go to the folder benhmark under intelai_models and use the command to run.

bash benchmark.sh framework={Framework} precision={Precision} model_name={Model name} mode={mode} data={Data path} checkpoint={checkpoint path} in_graph={pb file path} output_path={output path} infer_option={infer option} batch_size={batch size} steps={steps}


Recommended Parameters:
Framework = tensorflow
Precision = fp32
Model name = inceptionv3/bert_large/wide_deep_large_ds
Mode = Inference
infer_option =SQuAD [For Bert]

Arguments Specific to inceptionv3:
1.steps


Arguments specific to bert:
1.Data path
2.Checkpoint path
3.infer option


Arguments specific to Wide-deep:
1.Data path
