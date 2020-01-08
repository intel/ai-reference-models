#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from convert_tfrecords import ImageCoder, _process_image, _int64_feature, _float_feature, _bytes_feature, _bytes_list_feature


def load_annotation_data(annotations_filename):

    # Load annotation data
    with open(annotations_filename, 'r') as annotations_file:
        data = json.load(annotations_file)

    # Create map of category IDs to category names
    category_map = {}
    for category_datum in data['categories']:
        category_map[category_datum['id']] = category_datum['name']

    # Create map of file IDs to annotation data
    annotation_map = {}
    for annotation_datum in data['annotations']:
        image_id = annotation_datum['image_id']
        if (image_id not in annotation_map):
            annotation_map[image_id] = []

        # Add annotation datum for current image ID
        annotation_map[image_id].append(annotation_datum)

    # Create map of file IDs to image data
    image_map = {}
    for image_datum in data['images']:
        image_id = image_datum['id']
        if (image_id in annotation_map):
            image_map[image_id] = image_datum

    return image_map, annotation_map, category_map


def get_annotation_data(image_data, annotation_data, category_map):

    LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
                 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21,
                 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31,
                 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41,
                 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51,
                 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61,
                 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71,
                 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

    # Retrieve image width and height
    image_width = image_data['width']
    image_height = image_data['height']

    bboxes = []
    labels = []
    label_names = []
    difficult = []
    truncated = []
    for annotation_datum in annotation_data:
        # Scale bounding box coordinates
        # COCO bounding boxes are [x, y, width, height] but https://github.com/HiKapok/SSD.TensorFlow.git expects [ymin, xmin, ymax, xmax]
        bbox = annotation_datum['bbox']
        ymin = bbox[1] / image_height
        xmin = bbox[0] / image_width
        ymax = (bbox[1] + bbox[3]) / image_height
        xmax = (bbox[0] + bbox[2]) / image_width
        bboxes.append([ymin, xmin, ymax, xmax])

        labels.append(LABEL_MAP[annotation_datum['category_id']])
        label_names.append(category_map[annotation_datum['category_id']].encode('ascii'))

        # Append difficult and truncated flags
        difficult.append(0)
        truncated.append(0)

    return bboxes, labels, label_names, difficult, truncated


def get_record(filename, buffer, width, height, bboxes, labels, label_names, difficult, truncated):

    CHANNEL_COUNT = 3
    IMAGE_FORMAT = 'JPEG'

    # Extract bounding box coordinates
    ymin = []
    xmin = []
    ymax = []
    xmax = []
    for bbox in bboxes:
        ymin.append(bbox[0])
        xmin.append(bbox[1])
        ymax.append(bbox[2])
        xmax.append(bbox[3])

    # Create record features
    features = {
        'image/width': _int64_feature(width),
        'image/height': _int64_feature(height),
        'image/channels': _int64_feature(CHANNEL_COUNT),
        'image/shape': _int64_feature([height, width, CHANNEL_COUNT]),
        'image/object/bbox/xmin': _float_feature(xmin),
        'image/object/bbox/xmax': _float_feature(xmax),
        'image/object/bbox/ymin': _float_feature(ymin),
        'image/object/bbox/ymax': _float_feature(ymax),
        'image/object/bbox/label': _int64_feature(labels),
        'image/object/bbox/label_text': _bytes_list_feature(label_names),
        'image/object/bbox/difficult': _int64_feature(difficult),
        'image/object/bbox/truncated': _int64_feature(truncated),
        'image/format': _bytes_feature(IMAGE_FORMAT),
        'image/filename': _bytes_feature(filename.encode('utf8')),
        'image/encoded': _bytes_feature(buffer)}

    return tf.train.Example(features=tf.train.Features(feature=features))


def check_for_link(value):
    """
    Throws an error if the specified path is a link. os.islink returns
    True for sym links.  For files, we also look at the number of links in
    os.stat() to determine if it's a hard link.
    """
    if os.path.islink(value) or \
            (os.path.isfile(value) and os.stat(value).st_nlink > 1):
        raise argparse.ArgumentTypeError("{} cannot be a link.".format(value))


def check_valid_file_or_folder(value):
    """verifies filename exists and isn't a link"""
    if value is not None:
        if not os.path.isfile(value) and not os.path.isdir(value):
            raise argparse.ArgumentTypeError("{} does not exist or is not a file/folder.".
                                             format(value))
        check_for_link(value)
    return value


def main():

    RECORDS_PER_FILE = 1024
    RECORD_FILENAME_FORMAT = '%s-%.5d-of-%.5d'

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=check_valid_file_or_folder,
                        required=True, help='path to the input validation image files')
    parser.add_argument('--annotations_file', type=check_valid_file_or_folder,
                        required=True, help='name of the input validation annotations file')
    parser.add_argument('--output_prefix', type=str, required=True, help='prefix of the output TensorFlow record files')
    parser.add_argument('--output_path', type=check_valid_file_or_folder, required=True,
                        help='path to the output TensorFlow record files')

    args = parser.parse_args()

    # Load annotation data
    image_map, annotation_map, category_map = load_annotation_data(args.annotations_file)

    # Create output path if necessary
    if (not os.path.exists(args.output_path)):
        os.makedirs(args.output_path)

    # Create image coder
    image_coder = ImageCoder()

    record_file_index = 0
    record_file_count = np.ceil(len(image_map) / RECORDS_PER_FILE).astype(int)
    for index, image_id in tqdm(enumerate(image_map), desc='Generating', total=len(image_map), unit=' file'):
        # Create record writer
        if (index % RECORDS_PER_FILE == 0):
            output_filename = os.path.join(args.output_path, RECORD_FILENAME_FORMAT %
                                           (args.output_prefix, record_file_index, record_file_count))
            writer = tf.python_io.TFRecordWriter(output_filename)
            record_file_index += 1

        # Extract image data from current image file
        image_filename = image_map[image_id]['file_name']
        image_buffer, _, _ = _process_image(os.path.join(args.image_path, image_filename), image_coder)

        # Retrieve annotation data associated with current image file
        bboxes, labels, label_names, difficult, truncated = get_annotation_data(
            image_map[image_id], annotation_map[image_id], category_map)

        # Write TF record for current image file
        image_width, image_height = image_map[image_id]['width'], image_map[image_id]['height']
        record = get_record(image_filename, image_buffer, image_width, image_height,
                            bboxes, labels, label_names, difficult, truncated)
        writer.write(record.SerializeToString())


if __name__ == '__main__':

    main()
