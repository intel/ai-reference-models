<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`bfloat16_classifier_training.sh`](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/bfloat16/bfloat16_classifier_training.sh) | This script fine-tunes the bert base model on the Microsoft Research Paraphrase Corpus (MRPC) corpus, which only contains 3,600 examples. Download the [bert base uncased 12-layer, 768-hidden pretrained model](https://github.com/google-research/bert#pre-trained-models) and set the `CHECKPOINT_DIR` to that directory. The `DATASET_DIR` should point to the [GLUE data](#glue-data). |
| [`bfloat16_squad_training.sh`](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/bfloat16/bfloat16_squad_training.sh) | This script fine-tunes bert using SQuAD data. Download the [bert large uncased (whole word masking) pretrained model](https://github.com/google-research/bert#pre-trained-models) and set the `CHECKPOINT_DIR` to that directory. The `DATASET_DIR` should point to the [squad data files](#squad-data). |
| [`bfloat16_squad_training_demo.sh`](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/bfloat16/bfloat16_squad_training_demo.sh) | This script does a short demo run of 0.01 epochs using the `mini-dev-v1.1.json` file instead of the full SQuAD dataset. |
