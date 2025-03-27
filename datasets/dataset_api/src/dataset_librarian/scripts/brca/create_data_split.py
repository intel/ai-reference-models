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

import os
import shutil
import numpy as np
import argparse

from os import path, listdir
from pandas import DataFrame, read_csv
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from pathlib import Path

root_folder = os.environ.get("DATASET_DIR")


def get_subject_id(image_name):
    """
    Extracts the patient ID from an image filename.

    Args:
    - image_name: string representing the filename of an image

    Returns:
    - patient_id: string representing the patient ID extracted from the image filename
    """

    # Split the filename by "/"
    image_name = image_name.split("/")[-1]

    # Extract the first two substrings separated by "_", remove the first character (which is "P"), and join them
    # together to form the patient ID
    patient_id = "".join(image_name.split("_")[:2])[1:]

    return patient_id


def create_train_and_test_nlp(
    df: DataFrame, test_size: float
) -> Tuple[DataFrame, DataFrame]:
    """Split the dataset into training and testing sets for NLP.

    Args:
        df: Pandas DataFrame containing the data.
        test_size: Proportion of the dataset to include in the test split.

    Returns:
        Tuple of Pandas DataFrames for training and testing, respectively.
    """
    return train_test_split(df, test_size=test_size)


def copy_images(patient_ids: DataFrame, source_folder: str, target_folder: str) -> None:
    """Copy images of selected patients from the source folder to the target folder.

    Args:
        patient_ids: List of patient IDs for whom the images need to be copied.
        source_folder: Path to the source folder containing the images.
        target_folder: Path to the target folder where the images need to be copied.
    """

    for f in listdir(source_folder):
        if ("_CM_" in f) and (get_subject_id(f) in patient_ids.Patient_ID.to_list()):
            full_src_path = path.join(source_folder, f)
            shutil.copy(full_src_path, target_folder)


def create_train_and_test_vision_data(
    label_column,
    label_list,
    train_data,
    test_data,
    segmented_image_folder,
    target_folder,
):
    """
    Creates training and testing data for vision-based tasks by organizing segmented images based on labels.

    Args:
        label_column (str): The column name containing the class labels.
        label_list (list): List of unique class labels.
        train_data (pd.DataFrame): Training data containing the annotations.
        test_data (pd.DataFrame): Testing data containing the annotations.
        segmented_image_folder (str): Folder path containing segmented images.
        target_folder (str): Folder path to store the organized dataset.

    Returns:
        None
    """

    # Create the target directory for the dataset
    if path.exists(target_folder):
        shutil.rmtree(target_folder)

    # Iterate over train and test data
    for cat in ["test", "train"]:
        # Get the data based on the current category
        df = test_data if cat == "test" else train_data

        # Iterate over each label
        for label in label_list:
            # Source folder for images
            source_folder = path.join(segmented_image_folder, label)

            # Create target folder
            temp_target_folder = path.join(target_folder, cat, label)
            Path(temp_target_folder).mkdir(parents=True, exist_ok=True)

            # Copy images
            patient_ids = df[df[label_column] == label]
            copy_images(patient_ids, source_folder, temp_target_folder)


def data_preparation(
    annotation_file,
    target_annotation_folder,
    segmented_image_folder,
    target_image_folder,
    split_ratio,
):
    """
    Performs data preparation for breast cancer prediction, including creating training and testing data.

    Args:
        annotation_file (str): Location of the annotation file.
        target_annotation_folder (str): Location to save the processed annotation files.
        segmented_image_folder (str): Location of the folder containing segmented images.
        target_image_folder (str): Location to store the train and test images.
        split_ratio (float): Split ratio of the test data.

    Returns:
        None
    """

    input_data = read_csv(annotation_file)

    # create training and testing data
    training_data, testing_data = create_train_and_test_nlp(input_data, split_ratio)

    # get the path to training and testing data
    training_data_path = path.join(target_annotation_folder, "training_data.csv")
    testing_data_path = path.join(target_annotation_folder, "testing_data.csv")

    # save training and testing data
    training_data.to_csv(training_data_path, index=False)
    testing_data.to_csv(testing_data_path, index=False)

    # Get the list of class labels and the column containing the class label
    label_column = "label"
    label_list = np.sort(input_data[label_column].unique()).tolist()

    # create vision data
    create_train_and_test_vision_data(
        label_column,
        label_list,
        training_data,
        testing_data,
        segmented_image_folder,
        target_image_folder,
    )


if __name__ == "__main__":
    """
    Main entry point of the script for creating testing and training data for Breast Cancer prediction.
    Parses command-line arguments and calls the data_preparation function.

    Args:
        --annotation_file (str): Location of the annotation file.
        --target_annotation_folder (str): Location of the target annotation folder.
        --segmented_image_folder (str): Location of the segmented image folder.
        --target_image_folder (str): Location of the target image folder for train and test images.
        --split_ratio (float): Split ratio of the test data.

    Returns:
        None
    """

    parser = argparse.ArgumentParser(
        description="This function create testing and training data for Breast Cancer prediction."
    )

    parser.add_argument(
        "--annotation_file",
        type=str,
        help="Location of annotation file.",
        default=os.path.join(root_folder, "annotation", "annotation.csv"),
    )

    parser.add_argument(
        "--target_annotation_folder",
        type=str,
        help="Location of target annotation folder.",
        default=os.path.join(root_folder, "annotation"),
    )

    parser.add_argument(
        "--segmented_image_folder",
        type=str,
        help="Location of segmented image folder.",
        default=os.path.join(root_folder, "segmented_images"),
    )

    parser.add_argument(
        "--target_image_folder",
        type=str,
        help="Location of target imagefolder for train and test images.",
        default=os.path.join(root_folder, "train_test_split_images"),
    )

    parser.add_argument(
        "--split_ratio",
        type=float,
        help="split ratio of the test data.",
        default=0.1,
    )

    params = parser.parse_args()

    data_preparation(
        params.annotation_file,
        params.target_annotation_folder,
        params.segmented_image_folder,
        params.target_image_folder,
        params.split_ratio,
    )
