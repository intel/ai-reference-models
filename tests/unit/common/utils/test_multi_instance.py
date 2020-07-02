#!/usr/bin/env python
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
import os
import json
import unittest
from mock import patch

from common.utils.multi_instance import InferencePrefix, CPUInfo


class TestCase(unittest.TestCase):
    """The SenderFile tests."""

    @patch.object(CPUInfo, "_get_core_membind_info")
    def test_generate_inference_prefix_ht(self,
                                          get_core_membind_info_mock):
        """
        Test generate inference prefix HT.
        :param get_core_membind_info_mock: membind info mock
        :return: Nothing
        """
        tests_data_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(tests_data_dir, "files", "membind_info.json")) as json_data:
            membind_info = json.load(json_data)

        with open(os.path.join(tests_data_dir,
                               "files",
                               "inference_prefix_ht.json")) as json_data:
            ht_suite_data = json.load(json_data)

        inputs = []
        outputs = []
        for key, value in ht_suite_data.items():
            inputs.append(key.split(","))
            outputs.append(value)

        get_core_membind_info_mock.return_value = membind_info
        command = ["ls"]

        for i in range(len(inputs)):
            multi_instance = InferencePrefix(sockets=int(inputs[i][0]),
                                             instances=int(inputs[i][1]),
                                             cores_per_instance=int(inputs[i][2]))
            recived = multi_instance.generate_multi_instance_prefix(command, use_ht=True)
            self.assertEqual(outputs[i], recived, "Wrong generated prefix")

    @patch.object(CPUInfo, "_get_core_membind_info")
    def test_generate_inference_prefix_no_ht(self,
                                             get_core_membind_info_mock):
        """
        Test generate inference prefix HT.
        :param get_core_membind_info_mock: membind info mock
        :return: Nothing
        """
        tests_data_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(tests_data_dir, "files", "membind_info.json")) as json_data:
            membind_info = json.load(json_data)

        with open(os.path.join(tests_data_dir,
                               "files",
                               "inference_prefix_no_ht.json")) as json_data:
            ht_suite_data = json.load(json_data)

        inputs = []
        outputs = []
        for key, value in ht_suite_data.items():
            inputs.append(key.split(","))
            outputs.append(value)

        get_core_membind_info_mock.return_value = membind_info
        command = ["ls"]

        for i in range(len(inputs)):
            multi_instance = InferencePrefix(sockets=int(inputs[i][0]),
                                             instances=int(inputs[i][1]),
                                             cores_per_instance=int(inputs[i][2]))
            recived = multi_instance.generate_multi_instance_prefix(command, use_ht=False)
            self.assertEqual(outputs[i], recived, "Wrong generated prefix")

    @patch.object(CPUInfo, "_get_core_membind_info")
    def test_generate_inference_prefix_wrong_configuration(self,
                                                           get_core_membind_info_mock):
        """
        Test generate inference prefix HT.
        :param get_core_membind_info_mock: membind info mock
        :return: Nothing
        """
        tests_data_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(tests_data_dir, "files", "membind_info.json")) as json_data:
            membind_info = json.load(json_data)

        with open(os.path.join(tests_data_dir,
                               "files",
                               "inference_prefix_exception.json")) as json_data:
            ht_suite_data = json.load(json_data)

        inputs = []
        for item in ht_suite_data:
            inputs.append(item.split(","))

        get_core_membind_info_mock.return_value = membind_info
        command = ["ls"]

        for i in range(len(inputs)):
            multi_instance = InferencePrefix(sockets=int(inputs[i][0]),
                                             instances=int(inputs[i][1]),
                                             cores_per_instance=int(inputs[i][2]))
            with self.assertRaises(Exception):
                _ = multi_instance.generate_multi_instance_prefix(command, use_ht=False)
