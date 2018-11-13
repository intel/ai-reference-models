import argparse
import networks

_INTERVAL = 5000


class ArgParser(object):
    def __init__(self):
        self.parser = self._create_parser()

    def parse_args(self, args=None):
        args = self.parser.parse_args(args)
        return args

    @staticmethod
    def _create_parser():
        program_name = 'Squeezenet Training Program'
        desc = 'Program for training squeezenet with periodic evaluation.'
        parser = argparse.ArgumentParser(program_name, description=desc)

        parser.add_argument(
            '-f', "--inference-only",
            help='Only do inference evaluation.',
            dest='inference_only',
            action='store_true')
        parser.add_argument(
            '--verbose',
            default=False,
            action='store_true',
            help='''verbose logging.'''
        )
        parser.add_argument(
            '--cpu',
            default=True,
            action='store_true',
            help='''Use CPU or not.'''
        )
        parser.add_argument(
            '--model_dir',
            type=str,
            required=False,
            help='''Output directory for checkpoints and summaries.'''
        )
        parser.add_argument(
            '--data_location',  
            type=str,
            default='/datasets',  # must use absolute path, do not use ~
            required=False,
            help='''Directory of the TFRecords.'''
        )
        parser.add_argument(
            '--network',
            default='squeezenet_v11',
            type=str,
            required=False,
            choices=networks.catalogue
        )
        parser.add_argument(
            '--optimizer',
            default='momentum',
            type=str,
            required=False,
            choices=['momentum','adam']
        )
        parser.add_argument(
            '--target_image_size',
            default=[227, 227],
            nargs=2,
            type=int,
            help='''Input images will be resized to this.'''
        )
        parser.add_argument(
            '--num_classes',
            default=1001,  # has to be 1001 not 1000
            type=int,
            required=False,
            help='''Number of classes (unique labels) in the dataset.
                    Ignored if using CIFAR network version.'''
        )
        parser.add_argument(
            '--num_gpus',
            default=1,
            type=int,
            required=False,
        )
        parser.add_argument(
            '--batch_size',
            default=64,
            type=int,
            required=False
        )
        parser.add_argument(
            '--eval_steps',
            default=251,
            type=int,
            required=False
        )
        parser.add_argument(
            '--learning_rate', '-l',
            type=float,
            default=0.01,
            help='''Initial learning rate for ADAM optimizer.'''
        )
        parser.add_argument(
            '--weight_decay',  
            type=float,
            default=0.0,
            help='''L2 regularization factor for convolution layer weights.
                    0.0 indicates no regularization.'''
        )
        parser.add_argument(
            '--batch_norm_decay',
            type=float,
            default=0.9
        )
        parser.add_argument(
            '--decay_steps', 
            type=int,
            default=155000,
            help='''End learning rate for lr decay/cycling.'''
        )        
        parser.add_argument(
            '--min_lr', 
            type=float,
            default=0.000001,
            help='''End learning rate for lr decay/cycling.'''
        )
        parser.add_argument(
            '--num_input_threads',
            default=28,
            type=int,
            required=False,
            help='''The number input elements to process in parallel.'''
        )
        parser.add_argument(
            '--shuffle_buffer',
            default=512,
            type=int,
            required=False,
            help='''The minimum number of elements in the pool of training data
                    from which to randomly sample.'''
        )
        parser.add_argument(
            '--seed',
            default=1337,
            type=int
        )
        parser.add_argument(
            '--max_train_steps',
            default=199000,
            type=int
        )
        parser.add_argument(
            '--summary_interval',
            default=_INTERVAL,
            type=int
        )
        parser.add_argument(
            '--checkpoint_interval',
            default=5000,
            type=int
        )
        parser.add_argument(
            '--validation_interval',
            default= _INTERVAL,
            type=int
        )
        parser.add_argument(
            '--keep_last_n_checkpoints',
            default=10,
            type=int
        )
        parser.add_argument(
            '--timeline',
            default=0,
            type=int
        )
        parser.add_argument(
            '--KMP_BLOCKTIME',
            default=16,
            type=int
        )
        parser.add_argument(
            '--num_inter_threads',
            default=2,
            type=int
        )
        parser.add_argument(
            '--num_intra_threads',
            default=56,
            type=int
        )
        return parser
