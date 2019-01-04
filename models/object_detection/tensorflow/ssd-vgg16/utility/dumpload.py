#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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

import pickle
import numpy as np
import os


class DumpLoad:
    def __init__(self, pickle_filepath):
        self.pickle_filepath = pickle_filepath
        dir_path = os.path.dirname(pickle_filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        return
    def load(self):
        with open(self.pickle_filepath, 'rb') as handle:
            dataset = pickle.load(handle)
        return dataset
    def isExisiting(self):
        return os.path.exists(self.pickle_filepath) 
    def run(self):
        self.dump((np.arange(10), np.arange(20)))
        print(self.load())
        return
    
    def dump(self, dataset, protocoal = pickle.HIGHEST_PROTOCOL):
        with open(self.pickle_filepath, 'wb') as f:
            pickle.dump(dataset, f, protocoal)
        return
    

    
if __name__ == "__main__":   
    obj= DumpLoad('./data/in/in/myfile.pickle')
    obj.run()