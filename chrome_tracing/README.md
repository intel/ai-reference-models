# chrome tracing addition ot intel ai models
this folder contains files for inception-v3, bert_large, wide and deep models to which chrome tracing functionality has been added, there are no changes to runing commands to faciliate the storing of chrome tracing

inference.py -> file to replace for wide and deep under directory /intelai_models/models/recommendation/tensorflow/wide_deep_large_ds/inference/

eval_image_classifier_inference.py -> file to replace for inception-v3 under directory /intelai_models/models/image_recognition/tensorflow/inceptionv3/fp32/

run_squad.py -> file to replace for bert_large under directory /intelai_models/models/language_modeling/tensorflow/bert_large/inference/