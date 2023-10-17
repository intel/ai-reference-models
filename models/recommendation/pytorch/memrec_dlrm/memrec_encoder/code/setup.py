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

import platform
import sys

import numpy as np
from setuptools import Extension, setup

COMPILE_OPTIONS = []
LINK_OPTIONS = []


def is_new_osx():
    """Check whether we're on OSX >= 10.7"""
    if sys.platform != "darwin":
        return False
    mac_ver = platform.mac_ver()[0]
    if mac_ver.startswith("10"):
        minor_version = int(mac_ver.split(".")[1])
        return minor_version >= 7
    return False


if is_new_osx():
    # On Mac, use libc++ because Apple deprecated use of
    # libstdc
    COMPILE_OPTIONS.append("-stdlib=libc++")
    LINK_OPTIONS.append("-lc++")
    # g++ (used by unix compiler on mac) links to libstdc++ as a default lib.
    # See: https://stackoverflow.com/questions/1653047/avoid-linking-to-libstdc
    LINK_OPTIONS.append("-nodefaultlibs")

memrec_encodermodule = Extension(
    "memrec_encoder",
    sources=["memrec_encoder.cpp"],
    extra_compile_args=COMPILE_OPTIONS,
    extra_link_args=LINK_OPTIONS,
    include_dirs=[np.get_include()]
)

setup(
    name="memrec_encoder",
    version="3.0.0",
    description="Memrec embedding encoder module",
    ext_modules=[memrec_encodermodule],
    keywords="Memrec Embedding Encoder",
    long_description_content_type="text/markdown",
)
