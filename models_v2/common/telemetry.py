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

# System modules
import argparse
import fcntl
import os
import socket
import subprocess
import time
import fcntl
import socket
import sys
import select

platform_choices = [
    'Flex',
    'Max',
    'CUDA'
]

def read_output_to_file(file, process):
    if file == None:
        return
    
    # make reading non-blocking as we may not have output available
    flags = fcntl.fcntl(process.stdout, fcntl.F_GETFL) # get current p.stdout flags
    fcntl.fcntl(process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    with open(file, 'a') as f:
        try:
            while True:
                output = os.read(process.stdout.fileno(), 1024).decode('utf-8')
                f.write(output)
        except Exception as e:
            pass

def stop_ongoing_smi(smi_process, smi_file):
    if smi_file != None:
        read_output_to_file(smi_file, smi_process)
    if smi_process != None:
        smi_process.kill()
        smi_process = None

def start_smi(platform, output_dir):
    if platform in ['Flex', 'Max']:
        try:
            smi_process = subprocess.Popen('xpu-smi dump -d 0 -m 0,1,2,3,9,10,11,18,22,24,26,27,35,36,19,20'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False)                    
        except:
            print('error: "xpu-smi" for telemetry capture does not exist', file=sys.stderr)
            return None
    elif platform == 'CUDA':
        try:
            smi_process = subprocess.Popen([
                'nvidia-smi',
                '--query-gpu={0}'.format(','.join([
                    'timestamp', 'name', 'count', 'index', 'vbios_version', 'inforom.ecc',
                    'gpu_uuid', 'pci.bus_id', 'pci.device', 'pci.device_id', 'pci.sub_device_id',
                    'driver_version', 'pstate', 'pcie.link.gen.max', 'pcie.link.gen.current',
                    'pcie.link.gen.gpucurrent', 'pcie.link.gen.max', 'pcie.link.gen.gpumax',
                    'fan.speed', 'pstate', 'temperature.gpu', 'temperature.memory', 'utilization.gpu',
                    'utilization.memory', 'memory.total', 'memory.free', 'memory.used',
                    'compute_mode', 'encoder.stats.sessionCount', 'encoder.stats.averageFps',
                    'encoder.stats.averageLatency', 'ecc.mode.current', 'power.management',
                    'power.draw', 'power.limit', 'enforced.power.limit', 'power.default_limit',
                    'power.min_limit', 'power.max_limit', 'clocks.gr', 'clocks.sm', 'clocks.mem',
                    'clocks.video', 'clocks.max.gr', 'clocks.max.sm', 'clocks.max.mem'
                ])),
                '--format=csv', '-l', '1', '-i', '0',
                '-f', '{0}/nvidia_smi_dump.csv'.format(output_dir)
            ])
        except:
            print('error: "nvidia-smi" for telemetry capture does not exist', file=sys.stderr)
            return None
    return smi_process

def start(socket_path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    sock.sendall('start'.encode())
    sock.close()

def stop(socket_path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    sock.sendall('stop'.encode())
    sock.close()

def kill(socket_path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    sock.sendall('kill'.encode())
    sock.close()

def capture(args, output_dir=None):
    # fetch output location if a custom one wasn't provided
    if output_dir == None:
        output_dir = args.output_dir

    def terminate_capture(sock, conn, poller):
        if conn:
            conn.close()
        poller.unregister(sock)
        sock.close()

    if args.socket != '':
        try:
            os.unlink(args.socket)
        except OSError:
            if os.path.exists(args.socket):
                raise

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(args.socket)
        sock.listen(1)
        poller = select.poll()
        poller.register(sock, select.POLLIN)

    # begin telemetry capture with or without sample feedback
    smi_file = None
    if args.platform in ['Flex', 'Max']:
        # XPU SMI has no option to write output to a specific file.
        # As such we must pipe stdout to a file ourself.
        smi_file = os.path.join(output_dir, 'xpu_smi_dump.csv')
    smi_process = None
    has_ongoing_benchmark = False
    conn = None
    while True:
        msg = ''
        if args.socket != '':
            # Poll socket with timeout of 1 second:
            # - Prevents hangs in the event clients crashes.
            # - Allows for the process to incrementally do other work rather than just hanging until a message is received from client apps.
            events = poller.poll(1)
            if len(events):
                conn, addr = sock.accept()
                msg = conn.recv(1024).decode()
        else:
            time.sleep(1)
            msg = 'start'

        # check to do if termination was signaled through the socket
        if msg == 'kill':
            if has_ongoing_benchmark:
                stop_ongoing_smi(smi_process, smi_file)
                has_ongoing_benchmark = False
            if args.socket != '':
                terminate_capture(sock, conn, poller)
            return
        elif msg == 'start':
            if not has_ongoing_benchmark:
                smi_process = start_smi(args.platform, output_dir)
                if smi_process == None and args.socket != '':
                    terminate_capture(sock, conn, poller)
                    sys.exit(1)
                has_ongoing_benchmark = True
        elif msg == 'stop':
            if has_ongoing_benchmark:
                stop_ongoing_smi(smi_process, smi_file)
                has_ongoing_benchmark = False

        if has_ongoing_benchmark:
            # Dump process output to file every iteration of polling loop
            if smi_file != None:
                read_output_to_file(smi_file, smi_process)

        if args.socket != '' and conn:
            conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='telemetry',
          description='Capture telemetry with GPU SMI tool',
          epilog='Copyright (c) 2024 Intel Corporation')

    group = parser.add_argument_group('telemetry collection options')
    group.add_argument('--output_dir', action="store", type=str, default='', help='Path to store outputs')
    group.add_argument('--platform', help='System on which telemetry is being collected', default='Flex', choices=platform_choices)

    group1 = parser.add_argument_group('telemetry controls (require --socket)')
    group1.add_argument('--socket', action="store", type=str, default='', help='Socket to control telemetry capture')
    group1.add_argument('--start', action='store_true', help='Start telemetry capture')
    group1.add_argument('--stop', action='store_true', help='Stop telemetry capture')
    group1.add_argument('--kill', action='store_true', help='Kill telemetry process')

    args, benchmark_app_args = parser.parse_known_args()

    if args.start:
        start(args.socket)
        sys.exit(0)
    elif args.stop:
        stop(args.socket)
        sys.exit(0)
    elif args.kill:
        kill(args.socket)
        sys.exit(0)

    if args.output_dir == '':
        print('fatal: need output directory (--output_dir)', file=sys.stderr)
        sys.exit(1)

    # start telemetry capture in separate process
    capture(args)
