# Copyright 2018 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Creates an estimator to train the Transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tempfile
import random
import numpy.random


from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.python.training import adam

from mlperf_compliance import mlperf_log
from mlperf_compliance import tf_mlperf_log
from mlperf_utils.logs import hooks_helper

import compute_bleu
from data_download import VOCAB_FILE
from model import transformer
from model import model_params
import translate
from utils import dataset
from utils import metrics
from utils import tokenizer

tf.compat.v1.disable_eager_execution()
#Horovod support
global is_mpi 
try:
  import horovod.tensorflow as hvd
  hvd.init()
  is_mpi = hvd.size()
except ImportError:
  is_mpi = 0
  print("No MPI horovod support, this is running in no-MPI mode!")

DEFAULT_TRAIN_EPOCHS = 10
BLEU_DIR = "bleu"
INF = 10000


def model_fn(features, labels, mode, params):
  """Defines how to train, evaluate and predict from the transformer model."""
  with tf.compat.v1.variable_scope("model"):
    inputs, targets = features, labels

    # Create model and get output logits.
    model = transformer.Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)

    output = model(inputs, targets)

    # When in prediction mode, the labels/targets is None. The model output
    # is the prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=output)

    logits = output

    # Calculate model loss.
    xentropy, weights = metrics.padded_cross_entropy_loss(
        logits, targets, params.label_smoothing, params.vocab_size)
    loss = tf.reduce_sum(input_tensor=xentropy * weights) / tf.reduce_sum(input_tensor=weights)

    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, predictions={"predictions": logits},
          eval_metric_ops=metrics.get_eval_metrics(logits, labels, params))
    else:
      train_op = get_train_op(loss, params)
      logging_hook = tf.compat.v1.train.LoggingTensorHook({"loss" : loss }, every_n_iter=FLAGS.print_iter) 
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks = [logging_hook])


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
  """Calculate learning rate with linear warmup and rsqrt decay."""
  with tf.compat.v1.name_scope("learning_rate"):
    warmup_steps = tf.cast(learning_rate_warmup_steps, dtype=tf.float32)
    step = tf.cast(tf.compat.v1.train.get_or_create_global_step(), dtype=tf.float32)

    learning_rate *= (hidden_size ** -0.5)
    # Apply linear warmup
    learning_rate *= tf.minimum(1.0, step / warmup_steps)
    # Apply rsqrt decay
    learning_rate *= tf.math.rsqrt(tf.maximum(step, warmup_steps))

    # Save learning rate value to TensorBoard summary.
    tf.compat.v1.summary.scalar("learning_rate", learning_rate)

    return learning_rate


def get_train_op(loss, params):
  """Generate training operation that updates variables based on loss."""
  with tf.compat.v1.variable_scope("get_train_op"):
    mlperf_log.transformer_print(
        key=mlperf_log.OPT_LR_WARMUP_STEPS,
        value=params.learning_rate_warmup_steps)
    learning_rate = get_learning_rate(
        params.learning_rate, params.hidden_size,
        params.learning_rate_warmup_steps)
    log_id = mlperf_log.resnet_print(key=mlperf_log.OPT_LR, deferred=True)
    learning_rate = tf_mlperf_log.log_deferred(op=learning_rate, log_id=log_id,
                                               every_n=100)

    # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
    # than the TF core Adam optimizer.
    mlperf_log.transformer_print(key=mlperf_log.OPT_NAME,
                                 value=mlperf_log.LAZY_ADAM)
    mlperf_log.transformer_print(key=mlperf_log.OPT_HP_ADAM_BETA1,
                                 value=params.optimizer_adam_beta1)
    mlperf_log.transformer_print(key=mlperf_log.OPT_HP_ADAM_BETA2,
                                 value=params.optimizer_adam_beta2)
    mlperf_log.transformer_print(key=mlperf_log.OPT_HP_ADAM_EPSILON,
                                 value=params.optimizer_adam_epsilon)

    if is_mpi:
      learning_rate = learning_rate * hvd.size()
    # Using optimizer v1(from tensorflow.python.trainings*)
    # The optimizer v2 version of code is in the below. 
    # Optimzer v1 does not
    # have lazyAdam optimizer (was in contrib, now deprecated) 
    optimizer = adam.AdamOptimizer(
        learning_rate,
        beta1=params.optimizer_adam_beta1,
        beta2=params.optimizer_adam_beta2,
        epsilon=params.optimizer_adam_epsilon)

    if is_mpi:
      optimizer = hvd.DistributedOptimizer(optimizer)

    # Calculate and apply gradients using LazyAdamOptimizer.
    global_step = tf.compat.v1.train.get_global_step()
    tvars = tf.compat.v1.trainable_variables()
    grads_and_vars = optimizer.compute_gradients(
        loss, tvars)
    train_op = optimizer.apply_gradients(
        grads_and_vars,global_step=global_step, name="train")
    # Save gradient norm to Tensorboard
    tf.compat.v1.summary.scalar("global_norm/gradient_norm",
                      tf.linalg.global_norm(list(zip(*grads_and_vars))[0]))
    # Using tfa (tensorflow_addons) optimizer, which in turn 
    # uses optimizer_v2 (from tf.python.keras.optimizer_v2) 
    # which has waringin issues about global step not updated since
    # global_step is not accepted in apply_gradients() function of
    # optimizer_v2 version.
    # Thus the global step is updated and grouped with training op
    # To activate LazyAdams from tensroflow-addons activate the 
    # following code and take out the above optimer v1 related code
    # Currently both optimizer v1 and v2 take about same time
    '''                  
    optimizer = tfa.optimizers.LazyAdam(
        learning_rate,
        beta_1=params.optimizer_adam_beta1,
        beta_2=params.optimizer_adam_beta2,
        epsilon=params.optimizer_adam_epsilon)

    # Calculate and apply gradients using LazyAdamOptimizer.
    global_step = tf.compat.v1.train.get_global_step()
    tvars = tf.compat.v1.trainable_variables()
    tvars = tvars[0:len(tvars)-1]
    gradients = optimizer.get_gradients(
        loss, tvars)
    grads_and_vars = zip(gradients, tvars)
    train_op = optimizer.apply_gradients(
        grads_and_vars)
    # Save gradient norm to Tensorboard
    tf.compat.v1.summary.scalar("global_norm/gradient_norm",
                      tf.compat.v1.linalg.global_norm(list(gradients)))
    update_global_step = tf.compat.v1.assign(global_step, global_step + 1, name = "update_global_step")
    train_op = tf.compat.v1.group(train_op, [(update_global_step)])
    '''
    return train_op


def translate_and_compute_bleu(estimator, subtokenizer, bleu_source, bleu_ref):
  """Translate file and report the cased and uncased bleu scores."""
  # Create temporary file to store translation.
  tmp = tempfile.NamedTemporaryFile(delete=False)
  tmp_filename = tmp.name

  translate.translate_file(
      estimator, subtokenizer, bleu_source, output_file=tmp_filename,
      print_all_translations=False)

  # Compute uncased and cased bleu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
  os.remove(tmp_filename)
  return uncased_score, cased_score


def get_global_step(estimator):
  """Return estimator's last checkpoint."""
  return int(estimator.latest_checkpoint().split("-")[-1])


def evaluate_and_log_bleu(estimator, bleu_writer, bleu_source, bleu_ref):
  """Calculate and record the BLEU score."""
  subtokenizer = tokenizer.Subtokenizer(
      os.path.join(FLAGS.data_dir, FLAGS.vocab_file))

  uncased_score, cased_score = translate_and_compute_bleu(
      estimator, subtokenizer, bleu_source, bleu_ref)

  print("Bleu score (uncased):", uncased_score)
  print("Bleu score (cased):", cased_score)

  summary = tf.compat.v1.Summary(value=[
      tf.compat.v1.Summary.Value(tag="bleu/uncased", simple_value=uncased_score),
      tf.compat.v1.Summary.Value(tag="bleu/cased", simple_value=cased_score),
  ])

  bleu_writer.add_summary(summary, get_global_step(estimator))
  bleu_writer.flush()
  return uncased_score, cased_score


def train_schedule(
    estimator, train_eval_iterations, single_iteration_train_steps=None,
    single_iteration_train_epochs=None, bleu_source=None, bleu_ref=None,
    bleu_threshold=None):
  """Train and evaluate model, and optionally compute model's BLEU score.

  **Step vs. Epoch vs. Iteration**

  Steps and epochs are canonical terms used in TensorFlow and general machine
  learning. They are used to describe running a single process (train/eval):
    - Step refers to running the process through a single or batch of examples.
    - Epoch refers to running the process through an entire dataset.

  E.g. training a dataset with 100 examples. The dataset is
  divided into 20 batches with 5 examples per batch. A single training step
  trains the model on one batch. After 20 training steps, the model will have
  trained on every batch in the dataset, or, in other words, one epoch.

  Meanwhile, iteration is used in this implementation to describe running
  multiple processes (training and eval).
    - A single iteration:
      1. trains the model for a specific number of steps or epochs.
      2. evaluates the model.
      3. (if source and ref files are provided) compute BLEU score.

  This function runs through multiple train+eval+bleu iterations.

  Args:
    estimator: tf.Estimator containing model to train.
    train_eval_iterations: Number of times to repeat the train+eval iteration.
    single_iteration_train_steps: Number of steps to train in one iteration.
    single_iteration_train_epochs: Number of epochs to train in one iteration.
    bleu_source: File containing text to be translated for BLEU calculation.
    bleu_ref: File containing reference translations for BLEU calculation.
    bleu_threshold: minimum BLEU score before training is stopped.

  Raises:
    ValueError: if both or none of single_iteration_train_steps and
      single_iteration_train_epochs were defined.
  """
  # Ensure that exactly one of single_iteration_train_steps and
  # single_iteration_train_epochs is defined.
  if single_iteration_train_steps is None:
    if single_iteration_train_epochs is None:
      raise ValueError(
          "Exactly one of single_iteration_train_steps or "
          "single_iteration_train_epochs must be defined. Both were none.")
  else:
    if single_iteration_train_epochs is not None:
      raise ValueError(
          "Exactly one of single_iteration_train_steps or "
          "single_iteration_train_epochs must be defined. Both were defined.")

  evaluate_bleu = bleu_source is not None and bleu_ref is not None

  # Print out training schedule
  print("Training schedule:")
  if single_iteration_train_epochs is not None:
    print("\t1. Train for %d epochs." % single_iteration_train_epochs)
  else:
    print("\t1. Train for %d steps." % single_iteration_train_steps)
  print("\t2. Evaluate model.")
  if evaluate_bleu:
    print("\t3. Compute BLEU score.")
    if bleu_threshold is not None:
      print("Repeat above steps until the BLEU score reaches", bleu_threshold)
  if not evaluate_bleu or bleu_threshold is None:
    print("Repeat above steps %d times." % train_eval_iterations)

  if evaluate_bleu:
    # Set summary writer to log bleu score.
    bleu_writer = tf.compat.v1.summary.FileWriter(
        os.path.join(estimator.model_dir, BLEU_DIR))
    if bleu_threshold is not None:
      # Change loop stopping condition if bleu_threshold is defined.
      train_eval_iterations = INF

  # Loop training/evaluation/bleu cycles
  mlperf_log.transformer_print(key=mlperf_log.TRAIN_LOOP)
  # Profiling with timeline
  if FLAGS.save_profile == "Yes":
    profile_hooks = [tf.compat.v1.train.ProfilerHook(save_steps=1, output_dir=FLAGS.profile_dir)] # the json file 
  #profile file will be saved in in profile_dir
  #Creating hooks for printing Examples per Second, used with estimator.train
  training_batch_size = estimator.params.batch_size
  if FLAGS.batch_size != -1:
    training_batch_size = FLAGS.batch_size
  train_hooks = hooks_helper.get_train_hooks(
      ["ExamplesPerSecondHook"],
      model_dir=FLAGS.model_dir,
      batch_size=training_batch_size,
      every_n_steps=FLAGS.print_iter,
      warm_steps=50
  )
  if FLAGS.save_profile == "Yes":
    hooks = profile_hooks
  else:
    hooks = train_hooks
  
  for i in xrange(train_eval_iterations):
    print("Starting iteration", i + 1)

    if single_iteration_train_epochs is not None:
      mlperf_log.transformer_print(key=mlperf_log.TRAIN_EPOCH,
                                 value=i * single_iteration_train_epochs + 1)

    #Can we move the following out of the loop
    if is_mpi:
      train_hooks.append(hvd.BroadcastGlobalVariablesHook(0))
    # Train the model for single_iteration_train_steps or until the input fn
    # runs out of examples (if single_iteration_train_steps is None).
    estimator.train(dataset.train_input_fn, steps=single_iteration_train_steps, hooks=hooks)

    mlperf_log.transformer_print(key=mlperf_log.EVAL_START)
    # To save training time, we can turn off evaluation
    # Otherwise it will be turned on
    if FLAGS.do_eval == "Yes":
        eval_results = estimator.evaluate(dataset.eval_input_fn)
        print("Evaluation results (iter %d/%d):" % (i + 1, train_eval_iterations),
            eval_results)

    if evaluate_bleu:
      uncased_score, _ = evaluate_and_log_bleu(
          estimator, bleu_writer, bleu_source, bleu_ref)
      if bleu_threshold is not None and uncased_score > bleu_threshold:
        bleu_writer.close()
        break
      mlperf_log.transformer_print(key=mlperf_log.EVAL_TARGET, value=bleu_threshold)
      mlperf_log.transformer_print(key=mlperf_log.EVAL_ACCURACY, value=uncased_score)
    mlperf_log.transformer_print(key=mlperf_log.EVAL_STOP)


def main(_):
  # Set logging level to INFO to display training progress (logged by the
  # estimator)
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  mlperf_log.transformer_print(key=mlperf_log.RUN_START)

  #TODO: Do we need this, or remove
  if is_mpi:
    hvd.init()

  # Set random seed.
  if FLAGS.random_seed is None:
    raise Exception('No Random seed given')
  print('Setting random seed = ', FLAGS.random_seed)
  seed = FLAGS.random_seed
  mlperf_log.transformer_print(key=mlperf_log.RUN_SET_RANDOM_SEED, value=seed)
  random.seed(seed)
  tf.compat.v1.set_random_seed(seed)
  numpy.random.seed(seed)

  if FLAGS.params == "base":
    params = model_params.TransformerBaseParams
  elif FLAGS.params == "big":
    params = model_params.TransformerBigParams
  else:
    raise ValueError("Invalid parameter set defined: %s."
                     "Expected 'base' or 'big.'" % FLAGS.params)

  # Determine training schedule based on flags.
  if FLAGS.train_steps != 0 and FLAGS.train_epochs is not None:
    raise ValueError("Both --train_steps and --train_epochs were set. Only one "
                     "may be defined.")
  if FLAGS.train_steps != 0:
    train_eval_iterations = FLAGS.train_steps // FLAGS.steps_between_eval
    single_iteration_train_steps = FLAGS.steps_between_eval
    single_iteration_train_epochs = None
  else:
    if FLAGS.train_epochs is None:
      FLAGS.train_epochs = DEFAULT_TRAIN_EPOCHS
    train_eval_iterations = FLAGS.train_epochs // FLAGS.epochs_between_eval
    single_iteration_train_steps = None
    single_iteration_train_epochs = FLAGS.epochs_between_eval

  # Make sure that the BLEU source and ref files if set
  if FLAGS.bleu_source is not None and FLAGS.bleu_ref is not None:
    if not tf.io.gfile.exists(FLAGS.bleu_source):
      raise ValueError("BLEU source file %s does not exist" % FLAGS.bleu_source)
    if not tf.io.gfile.exists(FLAGS.bleu_ref):
      raise ValueError("BLEU source file %s does not exist" % FLAGS.bleu_ref)

  # Add flag-defined parameters to params object
  params.data_dir = FLAGS.data_dir
  params.num_cpu_cores = FLAGS.num_cpu_cores
  params.epochs_between_eval = FLAGS.epochs_between_eval
  params.repeat_dataset = single_iteration_train_epochs
  params.horovod = is_mpi 
  params.static_batch = FLAGS.static_batch
  # Add inter_op and intra_op parallelism thread
  session_config = tf.compat.v1.ConfigProto(
      inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
      intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
      allow_soft_placement=True)

  if FLAGS.save_checkpoints == "No":
    # To skip the checkpoints saving (which takes long time)  
    # added the following run_config
    run_config = tf.estimator.RunConfig(session_config=session_config,
       save_summary_steps=None, save_checkpoints_secs=None)
  else:
    run_config = tf.estimator.RunConfig(session_config=session_config)

  if is_mpi:
    FLAGS.model_dir = FLAGS.model_dir + str(hvd.rank())
  estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir, params=params, config=run_config)
  train_schedule(
      estimator, train_eval_iterations, single_iteration_train_steps,
      single_iteration_train_epochs, FLAGS.bleu_source, FLAGS.bleu_ref,
      FLAGS.bleu_threshold)

  mlperf_log.transformer_print(key=mlperf_log.RUN_STOP)
  mlperf_log.transformer_print(key=mlperf_log.RUN_FINAL)


if __name__ == "__main__":

  mlperf_log.ROOT_DIR_TRANSFORMER = os.path.normpath(
      os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--do_eval", "-de", type=str, default="No",
      help="[if evaluation needs to be done. If Yes, then do the evaluation"
           "if No, then the training time is faster.",
      metavar="<DE>")
  parser.add_argument(
      "--save_checkpoints", "-sc", type=str, default="No",
      help="[if checkpoints needs to be saved. If Yes, then save the checkpoints"
           "if No, then the training time is faster.",
      metavar="SC>")
  parser.add_argument(
      "--save_profile", "-sp", type=str, default="No",
      help="[if profile files(json) needs to be saved. If Yes, then save the profiles files"
           "if No, then the training time is faster.",
      metavar="SP>")
  parser.add_argument(
      "--data_dir", "-dd", type=str, default="/tmp/translate_ende",
      help="[default: %(default)s] Directory containing training and "
           "evaluation data, and vocab file used for encoding.",
      metavar="<DD>")
  parser.add_argument(
      "--vocab_file", "-vf", type=str, default=VOCAB_FILE,
      help="[default: %(default)s] Name of vocabulary file.",
      metavar="<vf>")
  parser.add_argument(
      "--model_dir", "-md", type=str, default="/tmp/fp32model",
      help="[default: %(default)s] Directory to save Transformer model "
           "training checkpoints",
      metavar="<MD>")
  parser.add_argument(
      "--params", "-p", type=str, default="big", choices=["base", "big"],
      help="[default: %(default)s] Parameter set to use when creating and "
           "training the model.",
      metavar="<P>")
  parser.add_argument(
      "--num_cpu_cores", "-nc", type=int, default=4,
      help="[default: %(default)s] Number of CPU cores to use in the input "
           "pipeline.",
      metavar="<NC>")

  # Flags for training with epochs. (default)
  parser.add_argument(
      "--train_epochs", "-te", type=int, default=None,
      help="The number of epochs used to train. If both --train_epochs and "
           "--train_steps are not set, the model will train for %d epochs." %
      DEFAULT_TRAIN_EPOCHS,
      metavar="<TE>")
  parser.add_argument(
      "--epochs_between_eval", "-ebe", type=int, default=1,
      help="[default: %(default)s] The number of training epochs to run "
           "between evaluations.",
      metavar="<TE>")

  # Flags for training with steps (may be used for debugging)
  parser.add_argument(
      "--train_steps", "-ts", type=int, default=None,
      help="Total number of training steps. If both --train_epochs and "
           "--train_steps are not set, the model will train for %d epochs." %
      DEFAULT_TRAIN_EPOCHS,
      metavar="<TS>")
  parser.add_argument(
      "--steps_between_eval", "-sbe", type=int, default=1000,
      help="[default: %(default)s] Number of training steps to run between "
           "evaluations.",
      metavar="<SBE>")

  # BLEU score computation
  parser.add_argument(
      "--bleu_source", "-bs", type=str, default=None,
      help="Path to source file containing text translate when calculating the "
           "official BLEU score. Both --bleu_source and --bleu_ref must be "
           "set. The BLEU score will be calculated during model evaluation.",
      metavar="<BS>")
  parser.add_argument(
      "--bleu_ref", "-br", type=str, default=None,
      help="Path to file containing the reference translation for calculating "
           "the official BLEU score. Both --bleu_source and --bleu_ref must be "
           "set. The BLEU score will be calculated during model evaluation.",
      metavar="<BR>")
  parser.add_argument(
      "--bleu_threshold", "-bt", type=float, default=None,
      help="Stop training when the uncased BLEU score reaches this value. "
           "Setting this overrides the total number of steps or epochs set by "
           "--train_steps or --train_epochs.",
      metavar="<BT>")
  parser.add_argument(
      "--random_seed", "-rs", type=int, default=None,
      help="the random seed to use", metavar="<SEED>")
  parser.add_argument(
      "--batch_size", "-bat", type=int, default=-1,
      help="change the training batch_size", metavar="<BATCH>")
  parser.add_argument(
      "--intra_op_parallelism_threads", "-intra", type=int, default=None,
      help="the intra op parallelism thread to use", metavar="<INTRA>")
  parser.add_argument(
      "--inter_op_parallelism_threads", "-inter", type=int, default=None,
      help="the intra op parallelism thread to use", metavar="<INTER>")
  parser.add_argument(
      "--print_iter", "-pi", type=int, default=None,
      help="print_iteration to print loss and timing", metavar="<PI>")
  # additional data_dir to save timeline
  parser.add_argument(
      "--profile_dir", "-od", type=str, default="/tmp/fp32profile",
      help="prifile dir", metavar="<OD>")
  parser.add_argument(
      "--static_batch", "-sb", type=str, default="No",
      help="Whether the batches in the dataset should have static shapes. In "
          "general, this setting should be False. Dynamic shapes allow the "
          "inputs to be grouped so that the number of padding tokens is "
          "minimized, and helps model training. In cases where the input shape "
          "must be static (e.g. running on TPU), this setting will be ignored "
          "and static batching will always be used.",
      metavar="<SB>")

  FLAGS, unparsed = parser.parse_known_args()

  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
