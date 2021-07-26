<!--- 30. Datasets -->
## Datasets

The <model name> accuracy scripts ([bfloat16_accuracy.sh](bfloat16_accuracy.sh)
and [bfloat16_accuracy_1200.sh](bfloat16_accuracy_1200.sh)) use the
[COCO validation dataset](http://cocodataset.org) in the TF records
format. See the [COCO dataset document](/datasets/coco/README.md) for
instructions on downloading and preprocessing the COCO validation dataset.

The performance benchmarking scripts ([bfloat16_inference.sh](bfloat16_inference.sh)
and [bfloat16_inference_1200.sh](bfloat16_inference_1200.sh)) use synthetic data,
so no dataset is required.
