date
export BERT_BASE_DIR=/nfs/site/home/jojimonv/Tensorflow/Models/Intel/BERT/uncased_L-12_H-768_A-12
export GLUE_DIR=/nfs/site/home/jojimonv/Tensorflow/Models/Intel/BERT/glue/glue_data
export TF_CPP_MIN_VLOG_LEVEL=5
export MKL_DNN_VERBOSE=1

echo "====================================================="
echo "    Running for $1...."
echo "====================================================="

python run_pretraining.py \
  --output_dir=./pretraining_output \
  --input_file=./output/tf_examples.tfrecord \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=2000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=False \
  --precision=$1 

#  --version_2_with_negative=True
date
