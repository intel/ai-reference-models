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

from os import path, listdir, makedirs
import shutil
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split
from collections import defaultdict

def copy_files_src_to_tgt(my_samples, fns_dict, my_src_folder, my_tgt_folder):
    """
    #copy files from source to target 
    """
    for ts in my_samples:
        files_to_copy = fns_dict.get(ts)
        for f in files_to_copy:
            src_fn = path.join(my_src_folder, f)
            tgt_fn = path.join(my_tgt_folder, f)
            shutil.copy2(src_fn, tgt_fn)
    return 

def create_data_split(infolder, outfolder):
    labels = listdir(infolder)
    print("Number of labels = ", len(labels))
    print("Labels are: \n", labels)
    for label in labels:
        fns = listdir(path.join(infolder, label))
        fns.sort()
        fns_root = ['_'.join(x.split('_')[:2]) for x in fns]
        # Convert list of tuples to dictionary value lists
        print("\nCreating default dict for stratifying the subject in {}.".format(label))
        fns_dict = defaultdict(list)
        for i, j in zip(fns_root, fns):
            fns_dict[i].append(j)
        train_samples, test_samples = train_test_split(list(fns_dict.keys()), test_size=0.2, random_state= 100)

        src_dir = path.join(infolder, label)
        tgt_dir = path.join(outfolder, 'train', label)
        makedirs(tgt_dir, exist_ok=True)
        copy_files_src_to_tgt(train_samples, fns_dict, src_dir, tgt_dir)

        tgt_dir = path.join(outfolder, 'test', label)
        makedirs(tgt_dir, exist_ok=True)
        copy_files_src_to_tgt(test_samples, fns_dict, src_dir, tgt_dir)

        print("Done splitting the files for label = {}\n".format(label))
    print("Done splitting the data. Output data is here: ", outfolder)
