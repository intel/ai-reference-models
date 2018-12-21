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

import numpy as np
import os
from nets.ssd import g_ssd_model


class EvalVOC:
    def __init__(self):
      
        
        return
    def __convert2np(self,net_outputs):
        temp = []
        for i in range(len(net_outputs)):
            net_output = net_outputs[i]
            
        return
    def eval_voc(self,image, filename,glabels,gbboxes,gdifficults,predictions, localizations):
        localizations = g_ssd_model.decode_bboxes_all_layers(localizations)
        localizations = localizations.reshape((localizations.shape[0], -1, localizations.shape[-1]))
        return
   
    def run(self):
        
        return
   
    
g_eval_voc = EvalVOC()
    
if __name__ == "__main__":   
    obj= EvalVOC()
    obj.run()