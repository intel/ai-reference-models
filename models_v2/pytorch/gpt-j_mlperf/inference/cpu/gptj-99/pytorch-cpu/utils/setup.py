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
# from distutils.core import setup, Extension
from setuptools import Extension
from setuptools import setup
from pathlib import Path
import os

dirname = Path(__file__).parent
#include_directories=os.path.join("include")
#print("!!!!!! {} !!!!!!!".format(include_directories))
module1 = Extension('thread_binder',
                    sources = ['thread_bind.cpp', 'kmp_launcher.cpp'],
                    depends=['kmp_launcher.hpp'],
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-lgomp']
                    )

setup (name = 'thread_binder',
       version = '0.11',
       description = 'Core binder for indepdendent threads',
       ext_modules = [module1])
