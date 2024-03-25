#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# AGPL-3.0 license
#


"""
COCO to YOLO Annotation Converter

This script processes annotations from a COCO-format JSON file and converts them into YOLO format text files.

Functionality:
- Reads COCO annotations from a specified JSON file.
- Creates a mapping of image dimensions for each image in the dataset.
- Converts COCO category IDs to a specific model's class IDs based on a predefined mapping.
- Normalizes bounding box coordinates (x_min, y_min, width, height) to relative coordinates (x_center, y_center, width, height) with respect to the dimensions of each image.
- Writes the converted annotations into separate text files for each image, following the YOLO format. Each line in a text file corresponds to an object in the image and contains class ID, x_center, y_center, width, and height (all normalized).
- Saves these text files in the 'data/labels/val2017' directory, creating the directory if it does not exist.

Usage:
- The script expects a path to a COCO-format JSON file as input.
- The output is a series of text files in the YOLO annotation format, one file per image in the dataset.

Note: 
- The script uses a predefined mapping (`coco_to_yolo_class_id`) to translate COCO category IDs to class IDs used in a specific model. This mapping might need adjustment based on the model's class definitions.
- The directory path 'data/labels/val2017' is hardcoded and may need to be modified according to the user's directory structure.
"""

import json
import os

def unpack_json_labels(json_file_path, root_dir):
    # Load COCO annotations
    with open(json_file_path) as f:
        data = json.load(f)

    # Create a mapping for image dimensions
    image_dimensions = {img['id']: (img['width'], img['height']) for img in data['images']}

    # Model's class ID mapping
    coco_to_yolo_class_id = {
        1: 0,   # person
        2: 1,   # bicycle
        3: 2,   # car
        4: 3,   # motorcycle
        5: 4,   # airplane
        6: 5,   # bus
        7: 6,   # train
        8: 7,   # truck
        9: 8,   # boat
        10: 9,  # traffic light
        11: 10, # fire hydrant
        13: 11, # stop sign
        14: 12, # parking meter
        15: 13, # bench
        16: 14, # bird
        17: 15, # cat
        18: 16, # dog
        19: 17, # horse
        20: 18, # sheep
        21: 19, # cow
        22: 20, # elephant
        23: 21, # bear
        24: 22, # zebra
        25: 23, # giraffe
        27: 24, # backpack
        28: 25, # umbrella
        31: 26, # handbag
        32: 27, # tie
        33: 28, # suitcase
        34: 29, # frisbee
        35: 30, # skis
        36: 31, # snowboard
        37: 32, # sports ball
        38: 33, # kite
        39: 34, # baseball bat
        40: 35, # baseball glove
        41: 36, # skateboard
        42: 37, # surfboard
        43: 38, # tennis racket
        44: 39, # bottle
        46: 40, # wine glass
        47: 41, # cup
        48: 42, # fork
        49: 43, # knife
        50: 44, # spoon
        51: 45, # bowl
        52: 46, # banana
        53: 47, # apple
        54: 48, # sandwich
        55: 49, # orange
        56: 50, # broccoli
        57: 51, # carrot
        58: 52, # hot dog
        59: 53, # pizza
        60: 54, # donut
        61: 55, # cake
        62: 56, # chair
        63: 57, # couch
        64: 58, # potted plant
        65: 59, # bed
        67: 60, # dining table
        70: 61, # toilet
        72: 62, # tv
        73: 63, # laptop
        74: 64, # mouse
        75: 65, # remote
        76: 66, # keyboard
        77: 67, # cell phone
        78: 68, # microwave
        79: 69, # oven
        80: 70, # toaster
        81: 71, # sink
        82: 72, # refrigerator
        84: 73, # book
        85: 74, # clock
        86: 75, # vase
        87: 76, # scissors
        88: 77, # teddy bear
        89: 78, # hair drier
        90: 79  # toothbrush
    }


    # Process each annotation
    for ann in data['annotations']:
        image_id = ann['image_id']
        file_name = f"{str(image_id).zfill(12)}.txt"
        dir_path = os.path.join(root_dir, 'data/labels/val2017')
        file_path = os.path.join(dir_path, file_name)
        os.makedirs(dir_path, exist_ok=True)

        # Convert COCO category ID to your model's class ID
        class_id = coco_to_yolo_class_id.get(ann['category_id'], -1)
        if class_id == -1:
            continue  # Skip if class ID not found in mapping

        # Get normalized bounding box coordinates
        x_min, y_min, width, height = ann['bbox']
        img_width, img_height = image_dimensions[image_id]
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        line = f"{class_id} {x_center} {y_center} {width} {height}\n"

        # Write to file
        with open(file_path, 'a') as file:
            file.write(line)
