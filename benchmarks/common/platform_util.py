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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform as system_platform
import subprocess

import sys

NUMA_NODES_STR_ = "NUMA node(s)"
CPU_SOCKETS_STR_ = "Socket(s)"
CORES_PER_SOCKET_STR_ = "Core(s) per socket"
THREADS_PER_CORE_STR_ = "Thread(s) per core"
LOGICAL_CPUS_STR_ = "CPU(s)"


class PlatformUtil:
    '''This module implements a platform utility that exposes functions that detect platform information.'''
    cpu_sockets_ = 0
    cores_per_socket_ = 0
    threads_per_core_ = 0
    logical_cpus_ = 0
    numa_nodes_ = 0

    def num_cpu_sockets(self):
        return self.cpu_sockets_

    def num_cores_per_socket(self):
        return self.cores_per_socket_

    def num_threads_per_core(self):
        return self.threads_per_core_

    def num_logical_cpus(self):
        return self.logical_cpus_

    def num_numa_nodes(self):
        return self.numa_nodes_

    def linux_init(self):
        # check to see if the lscpu command is present
        LSCPU_BIN = 'lscpu'
        lscpu_path = ''
        lscpu_path_cmd = "command -v lscpu"
        try:
            print("lscpu_path_cmd = {}".format(lscpu_path_cmd))
            lscpu_path = subprocess.check_output(lscpu_path_cmd, shell=True, stderr=subprocess.STDOUT).strip()
            print("lscpu located here: {}".format(lscpu_path))
            if not os.access(lscpu_path, os.F_OK | os.X_OK):
                raise ValueError("{} does not exist or is not executable.".format(lscpu_path))

            lscpu_output = subprocess.check_output([lscpu_path], stderr=subprocess.STDOUT)
            cpu_info = lscpu_output.decode('UTF-8').split('\n')

        except Exception as e:
            print("Problem getting CPU info: {}".format(e))
            sys.exit(1)

        # parse it
        for line in cpu_info:
            #      NUMA_NODES_STR_       = "NUMA node(s)"
            if line.find(NUMA_NODES_STR_) == 0:
                self.numa_nodes_ = int(line.split(":")[1].strip())
            #      CPU_SOCKETS_STR_      = "Socket(s)"
            elif line.find(CPU_SOCKETS_STR_) == 0:
                self.cpu_sockets_ = int(line.split(":")[1].strip())
            #      CORES_PER_SOCKET_STR_ = "Core(s) per socket"
            elif line.find(CORES_PER_SOCKET_STR_) == 0:
                self.cores_per_socket_ = int(line.split(":")[1].strip())
            #      THREADS_PER_CORE_STR_ = "Thread(s) per core"
            elif line.find(THREADS_PER_CORE_STR_) == 0:
                self.threads_per_core_ = int(line.split(":")[1].strip())
            #      LOGICAL_CPUS_STR_     = "CPU(s)"
            elif line.find(LOGICAL_CPUS_STR_) == 0:
                self.logical_cpus_ = int(line.split(":")[1].strip())

    def windows_init(self):
        raise NotImplementedError("Windows Support not yet implemented")

    def mac_init(self):
        raise NotImplementedError("Mac Support not yet implemented")

    def __init__(self):
        os_type = system_platform.system()
        if not os_type:
            raise ValueError("Unable to determine Operating system type.")
        elif "Windows" == os_type:
            self.windows_init()
        elif "Mac" == os_type or "Darwin" == os_type:
            self.mac_init()
        elif "Linux" == os_type:
            self.linux_init()
        else:
            raise ValueError("Unable to determine Operating system type.")
