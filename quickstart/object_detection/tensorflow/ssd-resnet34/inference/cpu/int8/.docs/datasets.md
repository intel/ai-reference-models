<!--- 30. Datasets -->
## Datasets

<model name> uses the [COCO dataset](https://cocodataset.org) for accuracy
testing.

Download and preprocess the COCO validation images using the
[instructions here](/datasets/coco). After the script to convert the raw
images to the TF records file completes, rename the tf_records file:
```
mv ${OUTPUT_DIR}/coco_val.record ${OUTPUT_DIR}/validation-00000-of-00001
```

Set the `DATASET_DIR` to the folder that has the `validation-00000-of-00001`
file when running the accuracy test. Note that the inference performance
test uses synthetic dataset.
