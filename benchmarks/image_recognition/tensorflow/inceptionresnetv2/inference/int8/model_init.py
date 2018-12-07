from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["KMP_SETTINGS"] = "1"


class ModelInitializer:
  """Detect the platform information and set necessary variables before launching the model"""

  def __init__(self, args, custom_args=[], platform_util=None):

    self.args = args
    self.platform_util = platform_util
    self.inference_command = ''

    # use default batch size if -1
    if self.args.batch_size == -1:
      self.args.batch_size = 128

    self.args.num_inter_threads = 1
    self.args.num_intra_threads = self.platform_util.num_cores_per_socket()

    if not self.args.single_socket:
      self.args.num_intra_threads *= self.platform_util.num_cpu_sockets()
      self.args.num_inter_threads = 2

    if self.args.benchmark_only:
      # benchmark_script = os.path.join(
      #   os.path.dirname(os.path.realpath(__file__)), "eval_image_classifier_benchmark.py")

      benchmark_script = os.path.join(self.args.intelai_models,
                                      self.args.platform, "eval_image_classifier_benchmark.py")
      self.inference_command = "python " + benchmark_script

      if self.args.single_socket:
        socket_id_str = str(self.args.socket_id)
        self.inference_command = \
          'numactl --cpunodebind=' + socket_id_str + ' --membind=' + socket_id_str + ' ' + self.inference_command

      os.environ["OMP_NUM_THREADS"] = str(self.args.num_intra_threads)

      self.inference_command = self.inference_command + \
                               ' --input-graph=' + self.args.input_graph + \
                               ' --inter-op-parallelism-threads=' + str(self.args.num_inter_threads) + \
                               ' --intra-op-parallelism-threads=' + str(self.args.num_intra_threads) + \
                               ' --batch-size=' + str(self.args.batch_size)

    elif self.args.accuracy_only:
      # accuracy_script = os.path.join(
      #   os.path.dirname(os.path.realpath(__file__)), "eval_image_classifier_accuracy.py")

      accuracy_script = os.path.join(self.args.intelai_models,
                                     self.args.platform, "eval_image_classifier_accuracy.py")
      self.inference_command = "python " + accuracy_script

      if self.args.single_socket:
        socket_id_str = str(self.args.socket_id)
        self.inference_command = \
          'numactl --cpunodebind=' + socket_id_str + ' --membind=' + socket_id_str + ' ' + self.inference_command

      os.environ["OMP_NUM_THREADS"] = str(self.args.num_intra_threads)

      self.inference_command = self.inference_command + \
                               ' --input_graph=' + self.args.input_graph + \
                               ' --data_location=' + self.args.data_location + \
                               ' --input_height=299' + \
                               ' --input_width=299' + \
                               ' --num_inter_threads=' + str(self.args.num_inter_threads) + \
                               ' --num_intra_threads=' + str(self.args.num_intra_threads) + \
                               ' --output_layer=InceptionResnetV2/Logits/Predictions' + \
                               ' --batch_size=' + str(self.args.batch_size)

    if self.args.verbose:
      print('Received these args: {}'.format(self.args))

  def run(self):
    """run command to enable model benchmark or accuracy measurement"""

    if self.inference_command:
      if self.args.verbose:
        print("Run model here.", self.inference_command)
      os.system(self.inference_command)
