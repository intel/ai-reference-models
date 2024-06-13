# Copyright (c) 2023-2024 Intel Corporation
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

# system modules
import torch.multiprocessing as mp
import time
import sys

# sample modules
import io_utils

def do_ipc_sync(barrier, sync_tag, terminate_if_sync_fail):
    if barrier.parties == 1:
        io_utils.write_info('Skipping IPC sync on tag {0} for single instance case.'.format(sync_tag))
        return

    io_utils.write_info('Doing IPC sync on tag: {0}...'.format(sync_tag))
    try:
        sync_start = time.time()
        barrier.wait()
        sync_end = time.time()
        io_utils.write_info('Sync on IPC tag {0} completed successfully in {1:.2f} seconds'.format(sync_tag, sync_end - sync_start))
    except mp.BrokenBarrierError:
        if terminate_if_sync_fail == True:
            io_utils.write_error('Sync on IPC tag {0} failed. Terminating.'.format(sync_tag))
            sys.exit(1)
        io_utils.write_warning('Sync on IPC tag {0} failed. Will continue execution.'.format(sync_tag))
        barrier.reset()
