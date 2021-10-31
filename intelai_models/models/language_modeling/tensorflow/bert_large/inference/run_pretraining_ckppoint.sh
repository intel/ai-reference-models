date
export BERT_BASE_DIR=/nfs/site/home/jojimonv/Tensorflow/Models/Intel/BERT/uncased_L-12_H-768_A-12
export GLUE_DIR=/nfs/site/home/jojimonv/Tensorflow/Models/Intel/BERT/glue/glue_data
export TF_CPP_MIN_VLOG_LEVEL=0
export MKL_DNN_VERBOSE=0

echo "====================================================="
echo "    Running for $1...."
echo "====================================================="

python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=./pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=False \
  --precision=$1 

#  --version_2_with_negative=True
date
