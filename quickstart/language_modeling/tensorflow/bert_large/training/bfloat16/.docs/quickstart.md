<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`bfloat16_classifier_training.sh`](bfloat16_classifier_training.sh) | This script fine-tunes the bert base model on the Microsoft Research Paraphrase Corpus (MRPC) corpus, which only contains 3,600 examples. Download the [bert base pretrained model](https://github.com/google-research/bert#pre-trained-models) and set the `CHECKPOINT_DIR` to that directory. The `DATASET_DIR` should point to the [GLUE data](#glue-data). |
| [`bfloat16_squad_training.sh`](bfloat16_squad_training.sh) | This script fine-tunes bert using SQuAD data. Download the [bert large pretrained model](https://github.com/google-research/bert#pre-trained-models) and set the `CHECKPOINT_DIR` to that directory. The `DATASET_DIR` should point to the [squad data files](#squad-data). |
| [`bfloat16_squad_training_demo.sh`](bfloat16_squad_training_demo.sh) | This script does a short demo run of 0.01 epochs using SQuAD data. |

These quickstart scripts can be run the following environments:
* [Bare metal](#bare-metal)
* [Docker](#docker)

