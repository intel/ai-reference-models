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
# SPDX-License-Identifier: EPL-2.0
#
import argparse
import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def convert_detection(label, detection):

	ID_INDEX = 0
	SCORE_INDEX = 1
	XMIN_INDEX = 2
	YMIN_INDEX = 3
	XMAX_INDEX = 4
	YMAX_INDEX = 5
	LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
		12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22,
		22: 23, 23: 24, 24: 25, 25: 27, 26: 28, 27: 31, 28: 32, 29: 33, 30: 34, 31: 35,
		32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46,
		42: 47, 43: 48, 44: 49, 45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55, 51: 56,
		52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62, 58: 63, 59: 64, 60: 65, 61: 67,
		62: 70, 63: 72, 64: 73, 65: 74, 66: 75, 67: 76, 68: 77, 69: 78, 70: 79, 71: 80,
		72: 81, 73: 82, 74: 84, 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90}

	# Extract image ID and bounding box score from detection
	image_id = int(detection[ID_INDEX])
	score = float(detection[SCORE_INDEX])

	# Convert bounding box coordinates [xmin, ymin, xmax, ymax] to [x, y, width, height]
	x = float(detection[XMIN_INDEX])
	y = float(detection[YMIN_INDEX])
	width = float(detection[XMAX_INDEX]) - x
	height = float(detection[YMAX_INDEX]) - y
	bbox = [x, y, width, height]

	return {'category_id': LABEL_MAP[label], 'image_id': image_id, 'score': score, 'bbox': bbox}


def generate_results_file(detections_path, results_filename):

	DETECTIONS_EXTENSION = '.txt'

	# Retrieve detections filenames
	filenames = [filename for filename in os.listdir(detections_path) if filename.endswith(DETECTIONS_EXTENSION)]

	results = []
	for filename in filenames:
		# Read detections from current file
		with open(os.path.join(detections_path, filename), 'r') as detections_file:
			lines = detections_file.readlines()

		# Convert detections from current file
		label = int(os.path.splitext(filename)[0].split('_')[1])
		for line in lines:
			results.append(convert_detection(label, line.strip().split()))

	# Write results to file
	with open(os.path.join(detections_path, results_filename), 'w') as results_file:
		json.dump(results, results_file)


def main():

	RESULTS_FILENAME = 'results.json'
	ANNOTATION_TYPE = 'bbox'

	parser = argparse.ArgumentParser()
	parser.add_argument('--detections_path', type = str, required = True, help = 'path to the input detected bounding box files')
	parser.add_argument('--annotations_file', type = str, required = True, help = 'name of the input validation annotations file')

	args = parser.parse_args()

	# Generate COCO results file
	print('Generating COCO results...')
	generate_results_file(args.detections_path, RESULTS_FILENAME)

	# Create COCO instance
	cocoGt = COCO(args.annotations_file)

	# Load COCO results
	cocoDt = cocoGt.loadRes(os.path.join(args.detections_path, RESULTS_FILENAME))

	# Evaluate results
	cocoEval = COCOeval(cocoGt, cocoDt, ANNOTATION_TYPE)
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()


if __name__ == '__main__':

	main()