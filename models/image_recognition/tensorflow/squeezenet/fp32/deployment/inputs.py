import tensorflow as tf
from tensorflow import data
import os,glob

_FILE_PATTERN = "%s*"
_SPLITS = set(["train", "validation"])
_R_MEAN = 123.68 / 255
_G_MEAN = 116.78 / 255
_B_MEAN = 103.94 / 255


def get_split(split_name, dataset_dir):
    """Gets a dataset tuple with instructions for reading ImageNet.
    
    Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    Returns:
    A python list.
    
    Raises:
    ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in _SPLITS:
        raise ValueError("split name %s was not recognized." % split_name)

    file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)
    files = glob.glob(file_pattern)
    if len(files) > 0:
        print (file_pattern, "Found %d %s files"%(len(files), split_name))
    else:
        raise IOError("zero %s file found in data directory"%(split_name))
    return files


class Pipeline(object):
    def __init__(self, args, sess):
        self.sess = sess
        self.batch_size = args.batch_size

        target_image_size = (args.target_image_size
                             if hasattr(args, 'target_image_size') else None)

        self._handle = tf.placeholder(tf.string, shape=[])
        self._is_training = tf.placeholder(tf.bool, [], 'is_training')

        if not args.inference_only:
            training_dataset = self._create_dataset(
                batch_size=args.batch_size * args.num_gpus,
                pad_batch=False,
                repeat=None,
                num_input_threads=args.num_input_threads,
                shuffle=True,
                shuffle_buffer=args.shuffle_buffer,
                seed=args.seed,
                files=get_split("train", args.data_location),
                distort_image=True,
                target_image_size=target_image_size
            )

            iterator = data.Iterator.from_string_handle(
                self._handle, training_dataset.output_types,
                training_dataset.output_shapes
            )
            self.data = iterator.get_next()
            training_iterator = training_dataset.make_one_shot_iterator()
            self._training_handle = sess.run(training_iterator.string_handle())
        else:
            validation_dataset = self._create_dataset(
                batch_size=args.batch_size * args.num_gpus,
                pad_batch=True,
                repeat=1,
                num_input_threads=args.num_input_threads,
                shuffle=False,
                shuffle_buffer=None,
                files=get_split("validation", args.data_location),
                distort_image=False,
                target_image_size=target_image_size
            )

            iterator = data.Iterator.from_string_handle(
                self._handle, validation_dataset.output_types,
                validation_dataset.output_shapes
            )
            self.data = iterator.get_next()
            self.validation_iterator = validation_dataset.\
                make_initializable_iterator()
            self.initialize_validation_data()
            self._validation_handle = sess.run(
                self.validation_iterator.string_handle())

    @property
    def is_training(self):
        return self._is_training

    @property
    def training_data(self):
        return {self._handle: self._training_handle,
                self._is_training: True}

    @property
    def validation_data(self):
        return {self._handle: self._validation_handle,
                self._is_training: False}

    def initialize_validation_data(self):
        self.sess.run(self.validation_iterator.initializer)

    @staticmethod
    def _create_dataset(batch_size,
                        pad_batch,
                        repeat,
                        num_input_threads,
                        shuffle,
                        shuffle_buffer,
                        files,
                        seed=1337,
                        distort_image=None,
                        target_image_size=None):
        assert batch_size % 2 == 0 or batch_size == 1

        input_processor = _InputProcessor(
            batch_size=batch_size,
            num_threads=num_input_threads,
            repeat=repeat,
            shuffle=shuffle,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            distort_image=distort_image,
            target_image_size=target_image_size
        )

        dataset = input_processor.from_tfrecords(files)
        if pad_batch:
            dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes=_get_padded_shapes(dataset.output_shapes, batch_size),
                padding_values=_get_padded_types(dataset.output_types)
            ).apply(tf.contrib.data.unbatch())
        return dataset


def _get_padded_shapes(output_shapes, batch_size):
    feature_shapes = dict()
    for feature, shape in output_shapes[0].items():
        feature_dims = shape.dims[1:]
        feature_shapes[feature] = tf.TensorShape(
            [tf.Dimension(batch_size)] + feature_dims)
    return feature_shapes, batch_size


def _get_padded_types(output_types):
    feature_values = dict()
    for feature, feature_type in output_types[0].items():
        feature_values[feature] = tf.constant(-1, feature_type)
    return feature_values, tf.constant(-1, tf.int64)


class _InputProcessor(object):
    def __init__(self,
                 batch_size,
                 num_threads,
                 repeat,
                 shuffle,
                 shuffle_buffer,
                 seed,
                 distort_image=None,
                 target_image_size=None):
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.repeat = repeat
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.distort_image = distort_image
        self.target_image_size = target_image_size

    def from_tfrecords(self, files):
        dataset = data.TFRecordDataset(files)
        dataset = dataset.map(
            map_func=self._preprocess_example,
            num_parallel_calls=self.num_threads
        )
        dataset = dataset.repeat(self.repeat)
        if self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer,
                seed=self.seed
            )
        dataset = dataset.batch(self.batch_size)
        return dataset

    def _preprocess_example(self, serialized_example):
        parsed_example = self._parse_serialized_example(serialized_example)
        image = self._preprocess_image(parsed_example['image/encoded'])
        return {'image': image}, parsed_example['image/class/label']

    def _preprocess_image(self, raw_image):
        image = tf.image.decode_jpeg(raw_image, channels=3)
        image = tf.image.resize_images(image, self.target_image_size)
        #image = tf.image.resize_image_with_crop_or_pad(image, 
        #                   self.target_image_size[0],
        #                    self.target_image_size[1],)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = self._mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        if self.distort_image:
            image = tf.image.random_flip_left_right(image)
        image = tf.transpose(image, [2, 0, 1])
        #image = tf.subtract(image, 0.5)
        #image = tf.multiply(image, 2.0)
        return image

    @staticmethod
    def _parse_serialized_example(serialized_example):
        features = {
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
        }
        return tf.parse_single_example(serialized=serialized_example,
                                       features=features)
    @staticmethod
    def _mean_image_subtraction(image, means):
        """Subtracts the given means from each image channel.

        For example:
            means = [123.68, 116.779, 103.939]
            image = _mean_image_subtraction(image, means)

        Note that the rank of `image` must be known.

        Args:
            image: a tensor of size [height, width, C].
            means: a C-vector of values to subtract from each channel.

        Returns:
            the centered image.

        Raises:
            ValueError: If the rank of `image` is unknown, if `image` has a rank other
            than three or if the number of channels in `image` doesn"t match the
            number of values in `means`.
        """
        if image.get_shape().ndims != 3:
            raise ValueError("Input must be of size [height, width, C>0]")
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError("len(means) must match the number of channels")

        channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=2, values=channels)