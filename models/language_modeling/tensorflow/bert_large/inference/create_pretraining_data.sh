date

if [ -z "${BERT_BASE_DIR}" ]; then
  echo "Please set the required bert directory as instructed."
  exit 1

export TF_CPP_MIN_VLOG_LEVEL=0
export MKL_DNN_VERBOSE=0

echo "====================================================="
echo "    Running create_pretraining.py for $1...."
echo "====================================================="
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=./output/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

date
