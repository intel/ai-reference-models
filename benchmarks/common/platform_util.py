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

#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform as system_platform
import subprocess
import sys

NUMA_NODES_STR_ = "NUMA node(s)"
CPU_SOCKETS_STR_ = "Socket(s)"
CORES_PER_SOCKET_STR_ = "Core(s) per socket"
THREADS_PER_CORE_STR_ = "Thread(s) per core"
LOGICAL_CPUS_STR_ = "CPU(s)"


class PlatformUtil:
    '''
    This module implements a platform utility that exposes functions that
    detects platform information.
    '''

    def __init__(self, args):
        self.args = args
        self.num_cpu_sockets = 0
        self.num_cores_per_socket = 0
        self.num_threads_per_core = 0
        self.num_logical_cpus = 0
        self.num_numa_nodes = 0

        os_type = system_platform.system()
        if "Windows" == os_type:
            self.windows_init()
        elif "Mac" == os_type or "Darwin" == os_type:
            self.mac_init()
        elif "Linux" == os_type:
            self.linux_init()
        else:
            raise ValueError("Unable to determine Operating system type.")

    def linux_init(self):
        lscpu_cmd = "lscpu"
        try:
            lscpu_output = subprocess.check_output([lscpu_cmd],
                                                   stderr=subprocess.STDOUT)
            # handle python2 vs 3 (bytes vs str type)
            if isinstance(lscpu_output, bytes):
                lscpu_output = lscpu_output.decode('utf-8')

            cpu_info = lscpu_output.split('\n')

        except Exception as e:
            print("Problem getting CPU info: {}".format(e))
            sys.exit(1)

        # parse it
        for line in cpu_info:
            #      NUMA_NODES_STR_       = "NUMA node(s)"
            if line.find(NUMA_NODES_STR_) == 0:
                self.num_numa_nodes = int(line.split(":")[1].strip())
            #      CPU_SOCKETS_STR_      = "Socket(s)"
            elif line.find(CPU_SOCKETS_STR_) == 0:
                self.num_cpu_sockets = int(line.split(":")[1].strip())
            #      CORES_PER_SOCKET_STR_ = "Core(s) per socket"
            elif line.find(CORES_PER_SOCKET_STR_) == 0:
                self.num_cores_per_socket = int(line.split(":")[1].strip())
            #      THREADS_PER_CORE_STR_ = "Thread(s) per core"
            elif line.find(THREADS_PER_CORE_STR_) == 0:
                self.num_threads_per_core = int(line.split(":")[1].strip())
            #      LOGICAL_CPUS_STR_     = "CPU(s)"
            elif line.find(LOGICAL_CPUS_STR_) == 0:
                self.num_logical_cpus = int(line.split(":")[1].strip())

    def windows_init(self):
        raise NotImplementedError("Windows Support not yet implemented")

    def mac_init(self):
        raise NotImplementedError("Mac Support not yet implemented")
