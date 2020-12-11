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

#

from argparse import ArgumentTypeError
import os
import re

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


def check_volume_mount(value):
    """
    Verifies that the value is a valid docker volume mount, where there should be
    at least two fields separated by a : (for the local directory to mount and the
    path to the where the directory will be mounted in the container. The third
    optional field is for extra options like read only.
    """
    if value:
        # Check that we have at least 2 fields and at most 3 fields
        if not 3 > value.count(":") > 0:
            raise ArgumentTypeError(
                "{} is not a valid volume mount string where ':' is used to separate the fields. "
                "See https://docs.docker.com/storage/volumes for information on formatting the volume "
                "mount string".format(value))

        # Check that the local directory specified is a valid folder and not a link
        check_valid_folder(value.split(':')[0])
    return value


def check_shm_size(value):
    """verfies the format of docker shm-size """
    if value is not None:
        if not re.match("([1-9][0-9]*)['b','k','m','g']", value):
            raise ArgumentTypeError("{} does not follow the --shm-size format definition.".format(value))
    return value


def check_num_cores_per_instance(value):
    """
    Verifies that the number of cores per instance is either 'socket' (specifying
    to use the number of cores per socket) or a positive integer.
    """
    if value:
        error_message = "Invalid number of cores per instance ({}). The value " \
                        "must be a positive integer or 'socket'".format(value)

        if value.lower().strip() == "socket":
            return "socket"
        elif not str(value).isdigit():
            raise ArgumentTypeError(error_message)

        try:
            check_positive_number(value)
        except ArgumentTypeError:
            raise ArgumentTypeError(error_message)
    return value
