from __future__ import print_function
import os


class ModelInitializer:
    """ SqueezeNet model initializer that calls train_squeezenet.py script
     from the models/image_recognition/tensorflow/squeezenet/fp32 directory"""

    def __init__(self, args, custom_args, platform_util):
        self.args = args
        self.custom_args = custom_args

        cores_per_socket = platform_util.num_cores_per_socket()
        self.args.num_inter_threads = 1
        self.args.num_intra_threads = cores_per_socket

        if not self.args.single_socket:
            self.args.num_intra_threads *= platform_util.num_cpu_sockets()

        if self.args.num_cores > 0:
            ncores = self.args.num_cores
        else:
            ncores = self.args.num_intra_threads

        script_path = os.path.join(self.args.intelai_models,
                                   self.args.platform, "train_squeezenet.py")

        self.command = ("taskset -c {:.0f}-{:.0f} python {} "
                        "--data_location {} --batch_size {:.0f} "
                        "--num_inter_threads {:.0f} --num_intra_threads {:.0f}"
                        " --model_dir {} --inference-only").format(
            self.args.socket_id * cores_per_socket,
            ncores - 1 + self.args.socket_id * cores_per_socket,
            script_path, self.args.data_location, self.args.batch_size,
            self.args.num_inter_threads, self.args.num_intra_threads,
            self.args.checkpoint)

        self.command += (' '.join(custom_args)).replace('\t', ' ')

    def run(self):
        if self.args.verbose:
            self.command += ' --verbose '
            print("Received these standard args: {}".format(self.args))
            print("Received these custom args: {}".format(self.custom_args))
            print(self.command)

        os.system(self.command)
