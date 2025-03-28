#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
# SPDX-License-Identifier: BSD-3-Clause

import os
import pandas as pd
import argparse
import shutil
from os import path
from PIL import Image, ImageDraw
import json

root_folder = os.environ.get('DATASET_DIR')

def classify_images(image_folder, annotation_file):
    print("Create folders for classified images")
    print(root_folder)
    benign = root_folder + "/vision_images/Benign"
    malignant =  root_folder + "/vision_images/Malignant"
    normal = root_folder + "/vision_images/Normal"
    os.makedirs(benign, exist_ok=True)
    os.makedirs(malignant, exist_ok=True)
    os.makedirs(normal, exist_ok=True)
    
    print("----- Classifying the images for data preprocessing -----")
    manual_annotations = pd.read_excel(annotation_file)
    print("Classify Low energy images")
    directory = image_folder + "/Low energy images of CDD-CESM"
    for filename in os.listdir(directory):
        src = os.path.join(directory, filename)
        # checking if it is a file   
        if os.path.isfile(src):
            patient_id = filename.strip(".jpg")
            classification_type = manual_annotations [manual_annotations ['Image_name'] == patient_id]["Pathology Classification/ Follow up"]
            if(classification_type.size != 0):
                tgt = root_folder + "/vision_images/" + str(classification_type.values[0])
                shutil.copy2(src, tgt)             
    print("Classify Subtracted energy images")
    directory = image_folder + "/Subtracted images of CDD-CESM"
    for filename in os.listdir(directory):
        src = os.path.join(directory, filename)
        # checking if it is a file   
        if os.path.isfile(src):
            patient_id = filename.strip(".jpg")
            classification_type = manual_annotations [manual_annotations ['Image_name'] == patient_id]["Pathology Classification/ Follow up"]
            if(classification_type.size != 0):
                tgt = root_folder + "/vision_images/" + str(classification_type.values[0])
                shutil.copy2(src, tgt)
    

def segment_images(segmentation_path,cesm_only = True):
    new_width = 512
    new_height = 512

    df = pd.read_csv(segmentation_path)

    # iterate over files in
    # Creating segmented images for Normal cases 
    directory = path.join(root_folder, "vision_images/Normal/")
    save_directory = path.join(root_folder, "segmented_images/Normal/")
    print(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Creating segmented images for \"Normal\" cases .........")
    for filename in os.listdir(directory):
        if cesm_only:
            if '_CM_' not in filename:
                continue
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            #print(f)
            new_file = os.path.join(save_directory, filename)
            im = Image.open(f)
            width, height = im.size   # Get dimensions
            
            cx, cy = width//2, height//2
            left, top = max(0, cx - new_width//2), max(0, cy - new_height//2) 
            right, bottom = min(width, cx + new_width//2), min(height, cy + new_height//2)

            # Crop the center of the image
            im = im.crop((left, top, right, bottom))
            im.save(new_file)

    # Creating segmented images for malignant cases 
    directory = path.join(root_folder, "vision_images/Malignant/")
    save_directory = path.join(root_folder, "segmented_images/Malignant/")
    os.makedirs(save_directory, exist_ok=True)
    print("Creating segmented images for \"malignant\" cases ..........")
    for filename in os.listdir(directory):
        if cesm_only:
            if '_CM_' not in filename:
                continue
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            rows = df.loc[df['#filename'] == filename]
            shapes = rows['region_shape_attributes'].values
            i = 0
            for shape in shapes:
                x = json.loads(shape)
                i = i + 1
                #print(filename)
                new_file = os.path.join(save_directory,  filename.split('.')[0]+str(i)+ '.' + filename.split('.')[1])
                #print(new_file)
                im = Image.open(f)
                width, height = im.size   # Get dimensions    
                if(x["name"] == "polygon"):
                    left = min(x["all_points_x"])
                    top = min(x["all_points_y"])
                    right = max(x["all_points_x"])
                    bottom = max(x["all_points_y"])
                    """
                    if (right - left) < new_width:
                        left, right = max(0, (right+left)//2 - new_width//2), min(width,  (right+left)//2 + new_width//2)
                    if (bottom - top) < new_height:
                        top, bottom = max(0, (top+bottom)//2 - new_height//2), min(height,  (top+bottom)//2 + new_height//2)
                    """
                    #print((left, top, right, bottom))
                    im = im.crop((left, top, right, bottom))
                    im.save(new_file)
                elif(x["name"] == "ellipse"):
                    #new_file = os.path.join(save_directory, str(i) + filename)
                    left = max(0, x["cx"] - new_width//2)
                    top = max(0, x["cy"] - new_height//2)
                    right = min(width, x["cx"] +  new_width//2)
                    bottom = min(height, x["cy"] + new_height//2)
                    im = im.crop((left, top, right, bottom))
                    im.save(new_file)

                elif(x["name"] == "circle"):
                    left = max(0, x["cx"] - new_width//2)
                    top = max(0, x["cy"] - new_height//2)
                    right = min(width, x["cx"] +  new_width//2)
                    bottom = min(height, x["cy"] + new_height//2)
                    im = im.crop((left, top, right, bottom))
                    im.save(new_file)



    # Creating segmented images for Benign cases
    directory = path.join(root_folder, "vision_images/Benign/")
    save_directory = path.join(root_folder, "segmented_images/Benign/")
    
    os.makedirs(save_directory, exist_ok=True)
    print("Creating segmented images for \"Benign\" cases ........")
    for filename in os.listdir(directory):
        if cesm_only:
            if '_CM_' not in filename:
                continue
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            rows = df.loc[df['#filename'] == filename]
            shapes = rows['region_shape_attributes'].values
            i = 0
            for shape in shapes:
                x = json.loads(shape)
                i = i + 1
                #print(filename)
                new_file = os.path.join(save_directory,  filename.split('.')[0]+str(i)+ '.' + filename.split('.')[1])
                #print(new_file)
                im = Image.open(f)
                width, height = im.size   # Get dimensions    
                if(x["name"] == "polygon"):
                    left = min(x["all_points_x"])
                    top = min(x["all_points_y"])
                    right = max(x["all_points_x"])
                    bottom = max(x["all_points_y"])
                    """
                    if (right - left) < new_width:
                        left, right = max(0, (right+left)//2 - new_width//2), min(width,  (right+left)//2 + new_width//2)
                    if (bottom - top) < new_height:
                        top, bottom = max(0, (top+bottom)//2 - new_height//2), min(height,  (top+bottom)//2 + new_height//2)
                    """
                    #print((left, top, right, bottom))
                    im = im.crop((left, top, right, bottom))
                    im.save(new_file)
                elif(x["name"] == "ellipse"):
                    #new_file = os.path.join(save_directory, str(i) + filename)
                    left = max(0, x["cx"] - new_width//2)
                    top = max(0, x["cy"] - new_height//2)
                    right = min(width, x["cx"] +  new_width//2)
                    bottom = min(height, x["cy"] + new_height//2)
                    im = im.crop((left, top, right, bottom))
                    im.save(new_file)

                elif(x["name"] == "circle"):
                    left = max(0, x["cx"] - new_width//2)
                    top = max(0, x["cy"] - new_height//2)
                    right = min(width, x["cx"] +  new_width//2)
                    bottom = min(height, x["cy"] + new_height//2)
                    im = im.crop((left, top, right, bottom))
                    im.save(new_file)
    return 0


def prepare_vision_data(image_folder, radiology_annotations_file, segmentation_path):
    #Classify images data into Benign,Malignant,Normal
    classify_images(image_folder, radiology_annotations_file)
    #Segment the classified images and save the ROI tiles
    segment_images(segmentation_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This function performs preprocessing steps for Breast Cancer annotation for vision images."
    )
    parser.add_argument(
        "--radiology_annotations_file",
        type=str,
        help="Location of manual annotations file",
        default=os.path.join(root_folder, "Radiology manual annotations.xlsx"),
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="Location of image folder",
        default=os.path.join(root_folder, "CDD-CESM"),
    )
    parser.add_argument(
        "--segmentation_path",
        type=str,
        help="Location of segmentation path",
        default=os.path.join(root_folder, "Radiology_hand_drawn_segmentations_v2.csv"),
    )

    params = parser.parse_args()
	#Preprocess data
    prepare_vision_data(
        params.image_folder,
        params.radiology_annotations_file,
        params.segmentation_path)
