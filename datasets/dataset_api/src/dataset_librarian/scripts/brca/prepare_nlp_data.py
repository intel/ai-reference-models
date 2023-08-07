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

import os
import docx2txt
import pandas as pd
import argparse
import zipfile
import shutil

root_folder = os.environ.get('DATASET_DIR')

def remove_item(item, s):  
    for c in range(s.count(item)):
        if item in s:
            str_indx = s.index(item)
            s = s[:str_indx] + s[str_indx+len(item)+3:]
            
    return s

def remove_BIRADS(df):
    symptoms_list = []
    for i, s in enumerate(df.symptoms): # [:10]:
        if '(BIRADS' in s:
            item = '(BIRADS'  
            s = remove_item(item, s)

        if '(BIRAD' in s:
            item = '(BIRAD'
            s = remove_item(item, s)
        
        symptoms_list.append(s)

    df.symptoms = symptoms_list
    
    return df

def read_right_and_left(tx):
    tx_right, tx_left = "", ""
    if "Right Breast:" in tx and "Left Breast:" in tx:
        tx = tx.split("Left Breast:")
        tx_right = [
            i
            for i in tx[0].split("Right Breast:")[1].splitlines()
            if ("ACR C:" not in i and i != "")
        ]
        tx_left = [i for i in tx[1].splitlines() if ("ACR C:" not in i and i != "")]

    elif "Right Breast:" in tx and "Left Breast:" not in tx:
        tx = tx.split("Right Breast:")[1].splitlines()
        tx_right = [i for i in tx if i != ""]

    elif "Right Breast:" not in tx and "Left Breast:" in tx:
        tx = tx.split("Left Breast:")[1].splitlines()
        tx_left = [i for i in tx if i != ""]

    return tx_right, tx_left


def read_content(file_content):

    annotation = file_content.split("OPINION:")  
    mm_revealed = annotation[0].split("REVEALED:")[1]
    mm_revealed_right, mm_revealed_left = read_right_and_left(mm_revealed)

    optinion = annotation[1].split("CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY REVEALED:")
    ces_mm_revealed = optinion[1]
    optinion = optinion[0]
    optinion_right, optinion_left = read_right_and_left(optinion)

    ces_mm_revealed_right, ces_mm_revealed_left = read_right_and_left(ces_mm_revealed)

    return (
        mm_revealed_right,
        mm_revealed_left,
        optinion_right,
        optinion_left,
        ces_mm_revealed_right,
        ces_mm_revealed_left,
    )


def add_df_log(df, dict_text, manual_annotations, f_id):
    for side in manual_annotations.Side.unique():
        for mm_type in manual_annotations.Type.unique().tolist() + ["OP"]:
            text_list = dict_text[mm_type + "_" + side]
            df_temp = manual_annotations[
                (manual_annotations.Patient_ID == int(f_id))
                & (manual_annotations.Side == side)
                & (manual_annotations.Type == mm_type)
            ]
            image_name = df_temp.Image_name.tolist()

            if mm_type == "OP":
                label = [None]
            else:
                label = df_temp["Pathology Classification/ Follow up"].unique().tolist()

            if len(label) == 1:
                df.loc[len(df)] = [
                    f_id,
                    image_name,
                    side,
                    mm_type,
                    label[0],
                    " ".join(text_list),
                ]

    return df


def label_correction(df):
    label_column = "label"
    data_column = "symptoms"
    patient_id = "Patient_ID"

    df_new = pd.DataFrame(columns=[label_column, data_column, patient_id])
    for i in df[patient_id].unique():
        annotation = " ".join(df[df[patient_id].isin([i])][data_column].to_list())
        temp_labels = [
            label_indx
            for label_indx in df[df[patient_id] == i][label_column].unique()
            if label_indx is not None
        ]

        if len(temp_labels) == 1:
            df_new.loc[len(df_new)] = [temp_labels[0], annotation, i]
        elif len(temp_labels) > 1:
            # CM images are substracted images, if available use the labels of the CM not DM
            # {patient number}_{breast side}_{image type}_{image view}; example ‘P1_L_CM_MLO’
            # (DM)   Digital mammography
            # (CESM) Contrast-enhanced spectral mammography

            df_temp = df[df[patient_id].isin([i])]

            if "CESM" in df_temp.Type.to_list():
                new_label = df_temp[df_temp.Type == "CESM"].label.to_list()[0]
                df_new.loc[len(df_new)] = [new_label, annotation, i]

        else:
            pass

    return df_new


def unzip_file(medical_reports_zip_file):
    medical_reports_folder = medical_reports_zip_file.split(".zip")[0].strip()

    if os.path.exists(medical_reports_folder):
        shutil.rmtree(medical_reports_folder)

    with zipfile.ZipFile(medical_reports_zip_file, "r") as zip_ref:
        zip_ref.extractall(root_folder)  # ( medical_reports_folder )

    return medical_reports_folder


def save_annotation_file(df, output_folder, file_name):
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(file_name, index=False)
    print("----- file is saved here :", file_name)


def prepare_data(
    medical_reports_zip_file,
    manual_annotations_file,
    output_annotations_folder,
    output_annotations_file,
):
    print("----- Starting the data preprocessing -----")
    manual_annotations = pd.read_excel(manual_annotations_file, sheet_name="all")

    medical_reports_folder = unzip_file(medical_reports_zip_file)

    df = pd.DataFrame(columns=["ID", "Image", "Side", "Type", "label", "symptoms"])
    
    for f in os.listdir(medical_reports_folder):
        DM_R, DM_L, OP_R, OP_L, CESM_R, CESM_L = "", "", "", "", "", ""
        f_id = f.split(".docx")[0].split("P")[1]

        try:
            file_content = docx2txt.process(os.path.join(medical_reports_folder, f))
        except Exception as e:
            Warning(e)

        DM_R, DM_L, OP_R, OP_L, CESM_R, CESM_L = read_content(file_content)
        dict_text = {
            "DM_R": DM_R,
            "DM_L": DM_L,
            "OP_R": OP_R,
            "OP_L": OP_L,
            "CESM_R": CESM_R,
            "CESM_L": CESM_L,
        }

        df = add_df_log(df, dict_text, manual_annotations, f_id)

    df["Patient_ID"] = [
        "".join([str(df.loc[i, "ID"]), df.loc[i, "Side"]]) for i in df.index
    ]

    df = label_correction(df)
    df = remove_BIRADS(df)

    save_annotation_file(df, output_annotations_folder, output_annotations_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This function performs preprocessing steps for Breast Cancer annotation."
    )

    parser.add_argument(
        "--medical_reports_folder",
        type=str,
        help="Location of medical reports for cases",
        default=os.path.join(root_folder, "Medical reports for cases .zip"),
    )
    parser.add_argument(
        "--manual_annotations_file",
        type=str,
        help="Location of manual annotations file",
        default=os.path.join(root_folder, "Radiology manual annotations.xlsx"),
    )
    parser.add_argument(
        "--output_annotations_folder",
        type=str,
        help="Location of output annotation folder",
        default=os.path.join(root_folder, "annotation"),
    )
    parser.add_argument(
        "--output_annotations_file",
        type=str,
        help="Name of the output annotation file",
        default=os.path.join(root_folder, "annotation", "annotation.csv"),
    )

    params = parser.parse_args()

    prepare_data(
        params.medical_reports_folder,
        params.manual_annotations_file,
        params.output_annotations_folder,
        params.output_annotations_file)
