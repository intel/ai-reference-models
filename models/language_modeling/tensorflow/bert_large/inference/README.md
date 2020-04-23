# BERT

**\*\*\*\*\* Sep 30th, 2019: Whole Word Masking Models \*\*\*\*\***

This document has instructions for how to run BERT for the 
following modes/platforms:
* [BFloat16 training](#training-instructions)

Changes from Google for new BERT model :

Supports bfloat16 based training on CPUs

## To run inferene

**build TF branch**
  `https://gitlab.devtools.intel.com/TensorFlow/Direct-Optimization/private-tensorflow/commits/amin/bert`

**Goto the dir**
  `cd models/language_modeling/tensorflow/bert_large/training/bfloat16`

**Run the following command**
`KMP_AFFINITY=granularity=fine,verbose,compact,1,0 KMP_BLOCKTIME=1 KMP_SETTINGS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=24 numactl -m 0 -c 0 python run_squad.py --vocab_file=$BERT_LARGE_DIR/vocab.txt --bert_config_file=$BERT_LARGE_DIR/bert_config.json --init_checkpoint=$SQUAD_CKPT_DIR/model.ckpt-7299 --do_predict=True --predict_file=$SQUAD_DATA_DIR/dev-v1.1.json --max_seq_length=384 --doc_stride=128 --output_dir=/tmp/squad-bfloat-out --mode=benchmark --predict_batch_size=32 --precision=bfloat16`


### Profile
`--mode=profile`

### Accuracy
`--mode=accuracy`
