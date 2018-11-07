import sys
import os

os.environ['PYTHONPATH'] = '$PYTHONPATH:.:./slim'
os.environ['PYTHONPATH'] = '$PYTHONPATH:.:./object_detection'
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0granularity=fine,compact,1,0'
os.environ['KMP_BLOCKTIME'] = '0'


class ModelInitializer:
  '''Add code here to detect the environment and set necessary variables before launching the model'''
  args=None
  custom_args=[]

  def run_inference_sanity_checks(self, args, custom_args):
      if args.input_graph == None:
            sys.exit('Please provide a path to the frozen graph directory via the \'--inference_graph\' flag.')
      if args.data_location == None:
            sys.exit('Please provide a path to the data directory via the \'--input_tfrecord_paths\' flag.')
      if not args.single_socket and args.num_cores == -1:
          print '***Warning***: Running inference on all cores could degrade performance. Pass \'--single-socket\' instead.\n'

  def __init__(self, args, custom_args, platform_util):
    self.args = args
    self.custom_args = custom_args
    platform_args = platform_util
    if self.args.verbose: 
      print('Received these standard args: {}'.format(self.args))
      print('Received these custom args: {}'.format(self.custom_args))
      print('Initialize here.')

    if args.mode == "inference":
	self.run_inference_sanity_checks(self.args, self.custom_args)

	if args.single_socket:
            args.num_inter_threads = 1
            args.num_intra_threads = platform_args.num_cores_per_socket()
        else:
            args.num_inter_threads = platform_args.num_cpu_sockets()
            if args.num_cores == -1:
                args.num_intra_threads = platform_args.num_cores_per_socket() * args.num_inter_threads
            else:
                args.num_intra_threads = args.num_cores

        cpuNumBegin = 0 if args.socket_id == 0 else (args.socket_id * platform_args.num_cores_per_socket())
        cpuNumEnd = (cpuNumBegin + args.num_intra_threads - 1) if args.num_cores == -1 else (cpuNumBegin + args.num_cores - 1)
        self.research_dir = os.path.join(args.model_source_dir, "research")
        self.run_cmd = "OMP_NUM_THREADS=" + str(args.num_intra_threads) + \
                       " numactl -l -N 1 " + \
                       "python object_detection/inference/infer_detections.py " + \
                       "--input_tfrecord_paths " + str(args.data_location) + \
                       " --inference_graph " + str(args.input_graph) + \
                       " --output_tfrecord_path=/tmp/ssd-mobilenet-record-out" + \
                       " --intra_op_parallelism_threads " + str(args.num_intra_threads) + \
                       " --inter_op_parallelism_threads " + str(args.num_inter_threads) + \
                       " --discard_image_pixels=True" + \
                       " --inference_only"  
    else:
            #TODO: Add training commands
            sys.exit("Training is currently not supported.") 

  def run(self):
    if self.args.verbose: print("Run model here.")
    original_dir = os.getcwd()
    os.chdir(self.research_dir)
    print("current directory: {}".format(os.getcwd()))
    print "Running: " + str(self.run_cmd)
    os.system(self.run_cmd)
    os.chdir(original_dir)

