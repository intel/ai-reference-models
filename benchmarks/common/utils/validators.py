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

from argparse import ArgumentTypeError
import os

"""
Functions used in `type=` arguments to ArgumentParser
Unfortunately can only take 1 string as argument https://docs.python.org/3/library/argparse.html#type
"""


def check_for_link(value):
    """
    Throws an error if the specified path is a link. os.islink returns
    True for sym links.  For files, we also look at the number of links in
    os.stat() to determine if it's a hard link.
    """
    if os.path.islink(value) or \
            (os.path.isfile(value) and os.stat(value).st_nlink > 1):
        raise ArgumentTypeError("{} cannot be a link.".format(value))


def check_no_spaces(value):
    """checks for spaces in string"""
    if ' ' in value:
        raise ArgumentTypeError("{} should not have whitespace(s).")
    return value


def check_positive_number(value):
    if value:
        value = int(value)
        if value <= 0:
            raise ArgumentTypeError("{} should be a positive number.")
    return value


def check_positive_number_or_equal_to_negative_one(value):
    if value:
        value = int(value)
        if value == 0 or value < -1:
            raise ArgumentTypeError("{} is not valid.".format(value))
    return value


def check_valid_filename(value):
    """verifies filename exists and isn't a link"""
    if value is not None:
        if not os.path.isfile(value):
            raise ArgumentTypeError("{} does not exist or is not a file.".
                                    format(value))
        check_for_link(value)
    return value


def check_valid_folder(value):
    """verifies filename exists and isn't a link"""
    if value is not None:
        if not os.path.isdir(value):
            raise ArgumentTypeError("{} does not exist or is not a directory.".
                                    format(value))
        check_for_link(value)
    return value


def check_valid_file_or_dir(value):
    """verfies file/dir exists and isn't a link"""
    if value is not None:
        if not os.path.exists(value):
            raise ArgumentTypeError("{} does not exist.".format(value))
        check_for_link(value)
    return value
