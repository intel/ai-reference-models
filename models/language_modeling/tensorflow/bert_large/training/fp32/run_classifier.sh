date
export BERT_BASE_DIR=/nfs/site/home/jojimonv/Tensorflow/Models/Intel/BERT/uncased_L-12_H-768_A-12
export GLUE_DIR=/nfs/site/home/jojimonv/Tensorflow/Models/Intel/BERT/glue/glue_data
export TF_CPP_MIN_VLOG_LEVEL=0
export MKL_DNN_VERBOSE=0

echo "====================================================="
echo "    Running for $1...."
echo "====================================================="
python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/nfs/site/home/jojimonv/Tensorflow/Models/Intel/BERT/bert/classify \
  --use_tpu=False \
  --precision=$1 \
  --do_lower_case=False 
#  --version_2_with_negative=True
date
