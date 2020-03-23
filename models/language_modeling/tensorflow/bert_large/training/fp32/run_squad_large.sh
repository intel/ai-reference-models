date
#export BERT_LARGE_DIR=/nfs/site/home/jojimonv/Tensorflow/Models/Intel/BERT/uncased_L-12_H-768_A-12
#export BERT_LARGE_DIR=/nfs/site/home/jojimonv/Tensorflow/Models/Intel/BERT/cased_L-24_H-1024_A-16
export BERT_LARGE_DIR=/nfs/site/home/jojimonv/Tensorflow/Models/Intel/BERT/wwm_cased_L-24_H-1024_A-16
export GLUE_DIR=/nfs/site/home/jojimonv/Tensorflow/Models/Intel/BERT/glue/glue_data
export SQUAD_DIR=/nfs/site/home/jojimonv/Tensorflow/Models/Intel/BERT/SQuAD
export TF_CPP_MIN_VLOG_LEVEL=0
#export MKLDNN_VERBOSE=3
#export MKL_DNN_VERBOSE=3
#export DNNL_VERBOSE=3
echo "====================================================="
echo "    Running for $1...."
echo "====================================================="
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=./large \
  --use_tpu=False \
  --precision=$1 \
  --do_lower_case=False 
#  --version_2_with_negative=True
date
