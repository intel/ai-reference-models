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
import glob
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile

def enabled(name: str, **kwargs):
    if name in kwargs and kwargs[name] is not None:
        return kwargs[name]
    if 'all' in kwargs:
        return kwargs['all']
    return False

def verbose(**kwargs):
    if 'verbose' in kwargs:
        return kwargs['verbose']
    return False

def quiet(**kwargs):
    if 'quiet' in kwargs:
        return kwargs['quiet']
    return False

def read_sysfs_file(file_name, **kwargs):
    try:
        with open(file_name, 'r') as file:
            return file.read().rstrip()
    except Exception as e:
        # Depending on kernel version some sysfs files might be absent,
        # so printing warning only in verbose mode
        if verbose(**kwargs):
            print('warning: ' + str(e), file=sys.stderr)
    return ''

def bytes_to(bytes, to):
    a = {
        'KB': 1000, 'MB': 1000**2, 'GB': 1000**3, 'TB': 1000**4,
        'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3, 'TiB': 1024**4,
    }
    return float(bytes) / float(a[to])

# Dictionary to store collected info from various components
# to avoid multi-calls of heavy functions.
_components_info = {}

# Cached status whether script is running under docker.
_is_docker = None

def is_docker():
    global _is_docker

    if _is_docker is None:
        if os.path.isfile('/.dockerenv') or os.path.isfile('/.dockerinit'):
            _is_docker = True
        else:
            _is_docker = False
    return _is_docker

def get_system():
    if is_docker():
        return 'docker0'
    else:
        return 'baremetal0'

def get_verified_pkg_info(pkg: str, **kwargs):
    global _components_info

    if 'dpkg' not in _components_info:
        return {}

    dpkg = _components_info['dpkg']['content']

    if not pkg in dpkg:
        return {}

    res = { pkg: dpkg[pkg] }

    if res[pkg]['integrity'] != 'unknown':
        return res

    try:
        subprocess.check_call(['dpkg', '--verify', pkg], stdout=subprocess.DEVNULL,
            stderr=None if verbose(**kwargs) else subprocess.DEVNULL)
        res[pkg]['integrity'] = 'ok'
    except Exception as e:
        res[pkg]['integrity'] = 'broken'

    return res

def get_verified_pkgs_info(pkgs: list, **kwargs):
    res = {}
    for p in pkgs:
        res |= get_verified_pkg_info(p, **kwargs)
    return res

def is_broken(pkgs: dict):
    for pkg in pkgs:
        if pkgs[pkg]['integrity'] == 'broken':
            return True
    return False

def get_target_pkg_list(target: str, **kwargs):
    try:
        result = subprocess.run(['dpkg', '-S', os.path.realpath(target)],
            check=True, stdout=subprocess.PIPE,
            stderr=None if verbose(**kwargs) else subprocess.DEVNULL)
    except Exception as e:
        if verbose(**kwargs):
            print('warning: ' + str(e), file=sys.stderr)
        return []

    owners = []
    results = result.stdout.decode().strip().split('\n')
    for result in results:
        result = result.replace(' ', '').split(':')[0]
        owners += result.split(',')

    return owners

# We are trying to implement the following paradigm to collect software system
# information.
# 1. "files" contain raw information collected by specific tools (like lshw,
#    svr-info, dpkg, etc.)
# 2. "config/system/*/software" contains curated subset of system information
#    which end-user considers relevant
#
# There are different options to control which information is collected (--cuda,
# --xpu, --dkms, --dpkg, etc.).
#
# Since there might be multiple sources of the same info, we follow this process
# to populate "config/system/*/software":
# 1. First, we try to identify whether software was installed via standard
#    installation methods (dpkg, rpm, pip, etc.).
# 2. If we find that software ingredients came from standard packages, we try
#    to verify packages consistency, i.e. that what we currently have on the
#    system is what package actually installs.
# 3. If all is clear package is getting reported to "config/system/*/software"
#    and is marked as not broken.
# 4. If package is not found or is found broken, we attempt to get raw information
#    on the software ingredients. Such software will be marked as installed by
#    "unknown" and marked "broken" if previously identified package was found
#    broken.

def get_cuda_info(**kwargs):
    global _components_info

    if 'cuda' not in _components_info:
        if not quiet(**kwargs):
            print('info: looking for cuda...', file=sys.stdout)

        try:
            with open('/usr/local/cuda/version.json', 'r') as f:
                _components_info['cuda'] = {
                    'file-type': 'cuda.version',
                    'system': get_system(),
                    'content': json.load(f)
                }
        except Exception as e:
            if verbose(**kwargs):
                print('warning: ' + str(e), file=sys.stderr)
            return {}

    if 'cuda' not in _components_info:
        return {}

    return { get_system() + '.cuda.version': _components_info['cuda'] }

def get_cuda_stack_info(**kwargs):
    global _components_info

    integrity = 'unknown'
    pkgs = get_target_pkg_list('/usr/local/cuda', **kwargs)

    res = get_verified_pkgs_info(pkgs, **kwargs)
    if res:
        if not is_broken(res):
            return res
        integrity = 'broken'

    if 'cuda' not in _components_info:
        return res

    res = _components_info['cuda']['content']
    for c in res:
        res[c]['installed-by'] = 'unknown'
        res[c]['integrity'] = integrity

    return res

def get_dkms_info(**kwargs):
    global _components_info

    if 'dkms' in _components_info:
        if 'dpkg' not in _components_info:
            return _components_info['dkms']

        dpkg = _components_info['dpkg']['content']

        res = {}
        for module in _components_info['dkms']:
            if module not in dpkg:
                res[module] = _components_info['dkms'][module]
                continue

            pkg = get_verified_pkg_info(module, **kwargs)
            if pkg[module]['integrity'] == 'broken' or _components_info['dkms'][module]['version'] not in dpkg[module]['version']:
                res[module] = _components_info['dkms'][module]
                res[module]['integrity'] = 'broken'
            else:
                res |= pkg

        return res

    if not quiet(**kwargs):
        print('info: calling dkms status...', file=sys.stdout)

    try:
        result = subprocess.run(['dkms', 'status', '-k', platform.release()],
            text=True, check=True, stdout=subprocess.PIPE,
            stderr=None if verbose(**kwargs) else subprocess.DEVNULL)
    except Exception as e:
        if verbose(**kwargs):
            print('warning: ' + str(e), file=sys.stderr)
        return {}

    res = {}
    for line in result.stdout.splitlines():
        # Example line: intel-i915-dkms/1.23.7.10.230608.12, 5.15.0-94-generic, x86_64: installed
        module = line.split('/')
        res[ module[0] ] = {
            'name': module[0],
            'version': module[1].split(',')[0],
            'installed-by': 'unknown',
            'integrity': 'unknown'
        }

    _components_info['dkms'] = res

    # call recursively to apply dpkg filter
    return get_dkms_info()

def get_docker_info(**kwargs):
    global _components_info

    if 'docker' not in _components_info:
        if not quiet(**kwargs):
            print('info: calling docker version..', file=sys.stdout)

        try:
            result = subprocess.run(['docker', 'version', '--format=json'],
                check=True, stdout=subprocess.PIPE,
                stderr=None if verbose(**kwargs) else subprocess.DEVNULL)
        except Exception as e:
            if verbose(**kwargs):
                print('warning: ' + str(e), file=sys.stderr)
            return {}

        _components_info['docker'] = {
            'file-type': 'docker.version',
            'system': get_system(),
            'content': json.loads(result.stdout)
        }

    if 'docker' not in _components_info:
        return {}

    return { get_system() + '.docker.version': _components_info['docker'] }

def get_docker_stack_info(**kwargs):
    global _components_info

    installed_by = 'unknown'
    integrity = 'unknown'

    res = get_verified_pkg_info('docker.io', **kwargs)
    if res:
        if not is_broken(res):
            return res
        installed_by = 'dpkg'
        integrity = 'broken'

    if 'docker' not in _components_info:
        return res

    docker_info = _components_info['docker']['content']

    for c in docker_info['Server']['Components']:
        if c['Name'] == 'Engine':
            engine = c

    res = {
        'docker-client': {
            'name': 'Docker Client',
            'version': docker_info['Client']['Version'],
            'installed-by': installed_by,
            'integrity': integrity
        },
        'docker-server': {
            'name': 'Docker Server',
            'version': engine['Version'],
            'installed-by': installed_by,
            'integrity': integrity
        }
    }
    return res

def get_xpu_stack_info(**kwargs):
    global _components_info

    if 'dpkg' not in _components_info:
        return {}

    packages = [
        'intel-cmemu',
        'intel-fw-gpu',
        'intel-gsc',
        'intel-gsc-dev',
        'intel-level-zero-gpu',
        'intel-level-zero-gpu-dev',
        'intel-media-va-driver',
        'intel-media-va-driver-non-free',
        'intel-metrics-discovery',
        'intel-metrics-discovery-dev',
        'intel-metrics-library',
        'intel-metrics-library-dev',
        'intel-microcode',
        'intel-oobmsm-dummy',
        'intel-opencl-icd',
        'level-zero',
        'level-zero-dev',
        'libdrm2',
        'libdrm-intel1',
        'libegl-mesa0',
        'libegl1-mesa',
        'libegl1-mesa-dev',
        'libgbm1',
        'libgl1-mesa-dev',
        'libgl1-mesa-dri',
        'libglapi-mesa',
        'libgles2-mesa-dev',
        'libglx-mesa0',
        'libigc-dev',
        'libigc1',
        'libigdfcl-dev',
        'libigdfcl1',
        'libigfxcmrt-dev',
        'libigfxcmrt7',
        'libmfx1',
        'libmfxgen1',
        'libvpl2',
        'libxatracker2',
        'mesa-va-drivers',
        'mesa-vdpau-drivers',
        'mesa-vulkan-drivers',
        'xpu-smi'
    ]

    packages.sort()

    dpkg = _components_info['dpkg']['content']
    res = {}
    for pkg in dpkg:
        for pattern in packages:
            if dpkg[pkg]['name'].startswith(pattern):
                res |= get_verified_pkg_info(pkg)

    return res

def get_dpkg_info(**kwargs):
    global _components_info

    if 'dpkg' in _components_info:
        return _components_info['dpkg']

    if not quiet(**kwargs):
        print('info: calling dpkg-query and dpkg --verify (might be time consuming)...', file=sys.stdout)

    dpkg = {
        'file-type': 'dpkg',
        'system': get_system(),
        'content': {}
    }

    try:
        result = subprocess.run(['dpkg-query', '--show',
            '-f={ "name": "${binary:Package}", "version": "${Version}", "status": "${Status}", "installed-by": "dpkg" }\n'],
            check=True, stdout=subprocess.PIPE,
            stderr=None if verbose(**kwargs) else subprocess.DEVNULL)
    except Exception as e:
        if verbose(**kwargs):
            print('warning: ' + str(e), file=sys.stderr)
            return {}

    results = result.stdout.decode().strip().split('\n')
    for result in results:
        result = json.loads(result)
        dpkg['content'][result['name']] = result
        dpkg['content'][result['name']]['integrity'] = 'unknown'

        if enabled('verify', **kwargs):
            try:
                subprocess.check_call(['dpkg', '--verify', result['name']], stdout=subprocess.DEVNULL,
                    stderr=None if verbose(**kwargs) else subprocess.DEVNULL)
                dpkg['content'][result['name']]['integrity'] = 'ok'
            except Exception as e:
                dpkg['content'][result['name']]['integrity'] = 'broken'

    _components_info['dpkg'] = dpkg

    return { get_system() + '.dpkg': _components_info['dpkg'] }

def get_lshw_info(**kwargs):
    global _components_info

    if 'lshw' not in _components_info:
        if not quiet(**kwargs):
            print('info: calling lshw (might be time consuming)...', file=sys.stdout)
        try:
            result = subprocess.run(['lshw', '-json'],
                check=True, stdout=subprocess.PIPE,
                stderr=None if verbose(**kwargs) else subprocess.DEVNULL)
        except Exception as e:
            if verbose(**kwargs):
                print('warning: ' + str(e), file=sys.stderr)
            return {}

        _components_info['lshw'] = {
            'file-type': 'lshw',
            'system': get_system(),
            'content': json.loads(result.stdout)
        }

    if 'lshw' not in _components_info:
        return {}

    return { get_system() + '.lshw': _components_info['lshw'] }

def get_pytorch_info(**kwargs):
    if not quiet(**kwargs):
        print('info: importing PyTorch components (might be time consuming)...', file=sys.stdout)
    try:
        import torch
    except Exception as e:
        if verbose(**kwargs):
            print('warning: ' + str(e), file=sys.stderr)
            print('warning: cannot import \'torch\'', file=sys.stderr)
        return {}

    res = {
        'pytorch': {
            'name': 'PyTorch',
            'version': str(torch.__version__)
        }
    }

    try:
        import intel_extension_for_pytorch as ipex
        res['ipex'] = {
            'name': 'IPEX',
            'version': str(ipex.__version__),
            'has_onemkl': ipex.xpu.has_onemkl()
        }
    except Exception as e:
        if verbose(**kwargs):
            print('warning: ' + str(e), file=sys.stderr)
            print('warning: cannot import \'intel_extension_for_pytorch\'', file=sys.stderr)

    return res

def get_svrinfo(**kwargs):
    global _components_info

    if 'svrinfo' not in _components_info:
        if not quiet(**kwargs):
            print('info: calling svr-info (might be time consuming)...', file=sys.stdout)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                result = subprocess.run(['svr-info', '-format', 'json', '-output', tmpdir],
                    check=True,
                    stdout=None if quiet(**kwargs) else subprocess.DEVNULL,
                    stderr=None if verbose(**kwargs) else subprocess.DEVNULL)
                files = glob.glob(os.path.join(tmpdir, '*.json'))
                with open(files[0]) as file:
                    _components_info['svrinfo'] = {
                        'file-type': 'svrinfo',
                        'system': get_system(),
                        'content': json.load(file)
                    }
            except Exception as e:
                if verbose(**kwargs):
                    print('warning: ' + str(e), file=sys.stderr)
                return {}

    if 'svrinfo' not in _components_info:
        return {}

    return { get_system() + '.svrinfo': _components_info['svrinfo'] }

def get_software(**kwargs):
    res = {
        'os': {
            'name': platform.freedesktop_os_release()['NAME'],
            'version': platform.freedesktop_os_release()['VERSION']
        },
        'kernel': {
            'name': 'kernel',
            'version': platform.release()
        },
        'python': {
            'name': 'Python',
            'version': platform.python_version()
        },
    }

    if enabled('cuda', **kwargs):
        res |= get_cuda_stack_info(**kwargs)

    if enabled('dkms', **kwargs):
        res |= get_dkms_info(**kwargs)

    if enabled('docker', **kwargs):
        res |= get_docker_stack_info(**kwargs)

    if enabled('pytorch', **kwargs):
        res |= get_pytorch_info(**kwargs)

    if enabled('xpu', **kwargs):
        res |= get_xpu_stack_info(**kwargs)

    return res

def get_i915_prelim(card, **kwargs):
    prelim_uapi_version = read_sysfs_file(card + '/prelim_uapi_version', **kwargs)
    if prelim_uapi_version == '':
        return {}
    res = {
        'prelim.uapi.version': prelim_uapi_version
    }
    return res

def get_gpu_info(card, **kwargs):
    res = {}

    pci_id = read_sysfs_file(card + '/device/device')
    vendor_id = read_sysfs_file(card + '/device/vendor')

    if vendor_id.lower() == '0x10de'.lower():
        res |= {
            'name': pci_id,
            'vendor': 'NVIDIA Corporation'
        }
        if pci_id.lower() == '0x2236'.lower():
            res |= {
                'name': 'GA102GL [A10]',
                'link': 'https://www.nvidia.com/en-us/data-center/products/a10-gpu/'
            }
        elif pci_id.lower() == '0x26b9'.lower():
            res |= {
                'name': 'L40',
                'link': 'https://www.nvidia.com/en-us/data-center/l40/'
            }
        elif pci_id.lower() == '0x27b8'.lower():
            res |= {
                'name': 'L4',
                'link': 'https://www.nvidia.com/en-us/data-center/l4/'
            }
    elif vendor_id.lower() == '0x1a03'.lower():
        res |= {
            'name': pci_id,
            'vendor': 'ASPEED Technology'
        }
    elif vendor_id.lower() == '0x8086'.lower():
        res |= {
            'vendor': 'Intel Corporation',
            'name': pci_id  # will rewrite below if we know what device is that
            }

        if pci_id.lower() == '0x56c1'.lower():
            res |= {
                'name': 'Intel® Data Center GPU Flex 140',
                'link': 'https://ark.intel.com/content/www/us/en/ark/products/230020/intel-data-center-gpu-flex-140.html'
            }
        elif pci_id.lower() == '0x56c0'.lower():
            res |= {
                'name' : 'Intel® Data Center GPU Flex 170',
                'link': 'https://ark.intel.com/content/www/us/en/ark/products/230019/intel-data-center-gpu-flex-170.html'
            }

        res |= {
            'frequency.min': {
                'value': int(read_sysfs_file(card + '/gt_min_freq_mhz', **kwargs)),
                'units': 'MHz'
            }
        }
        res |= {
            'frequency.max': {
                'value': int(read_sysfs_file(card + '/gt_max_freq_mhz', **kwargs)),
                'units': 'MHz'
            }
        }
        res |= {
            'frequency.boost': {
                'value': int(read_sysfs_file(card + '/gt_boost_freq_mhz', **kwargs)),
                'units': 'MHz'
            }
        }
        # Get i915 prelim uAPI fields (prelim uAPI is available in Intel custom kernels, absent in vanilla kernel)
        res |= get_i915_prelim(card, **kwargs)
    else:
        res |= {
            'name': pci_id,
            'vendor': vendor_id
        }

    return res

def lshw_get_desc(lshw_info, id):
    if 'children' not in lshw_info:
        return {}
    for item in lshw_info['children']:
        if item['id'] == id:
            return item
        res = lshw_get_desc(item, id)
        if res:
            return res
    return {}

def lshw_get_ids(lshw_info, pattern):
    if 'children' not in lshw_info:
        return []
    res = []
    for item in lshw_info['children']:
        if re.match(pattern, item['id']):
            res += [ item['id'] ]
        child_ids = lshw_get_ids(item, pattern)
        if child_ids:
            res += child_ids
    return res

def get_cpu_info(**kwargs):
    res = {}
    if 'lshw' in _components_info:
        cpus = lshw_get_ids(_components_info['lshw']['content'], '^cpu')
        for cpu in cpus:
            cpu_info = lshw_get_desc(_components_info['lshw']['content'], cpu)

            # Placeholder for the case of empty cpu socket
            cpu_res = { 'name': 'Empty socket', 'vendor': 'Empty socket' }

            if 'product' in cpu_info and 'vendor' in cpu_info:
                # Filling in info we would get running lshw from non-privileged user
                cpu_res |= {
                    'name': cpu_info['product'],
                    'vendor': cpu_info['vendor']
                }

                cpu_list = os.listdir('/sys/devices/system/cpu/')
                cpu_list.sort()
                config = {}
                for c in cpu_list:
                    if not re.match('^cpu[0-9]*$', c):
                        continue

                    path = '/sys/devices/system/cpu/' + c
                    node = glob.glob(path + '/node*')
                    if cpu != 'cpu' and cpu != re.sub('.*/node', 'cpu:', node[0]):
                        continue

                    driver = read_sysfs_file(path + '/cpufreq/scaling_driver', **kwargs)
                    governor = read_sysfs_file(path + '/cpufreq/scaling_governor', **kwargs)
                    min_frequency = read_sysfs_file(path + '/cpufreq/scaling_min_freq', **kwargs)
                    max_frequency = read_sysfs_file(path + '/cpufreq/scaling_max_freq', **kwargs)

                    def pretty_freq(freq):
                        return { 'value': int(float(freq)/float(1000)), 'units': 'MHz' }

                    config[c] = {
                        'frequency.driver': driver,
                        'frequency.governor': governor,
                        'frequency.min': pretty_freq(min_frequency),
                        'frequency.max': pretty_freq(max_frequency)
                    }
                same_config = True
                c0 = ''
                for c in config:
                    if c0 == '':
                        c0 = c
                    if config[c] != config[c0]:
                        same_config = False
                if same_config:
                    cpu_res |= { 'configuration': config[c0] }
                else:
                    cpu_res |= { 'configuration': { 'cpus': config } }

                # Going into details if lshw was run from privileged user
                if 'configuration' in cpu_info:
                    for c in cpu_info['configuration']:
                        if c == 'cores':
                            cpu_res['configuration'] |= { 'cores.total': int(cpu_info['configuration'][c]) }
                        elif c == 'enabledcores':
                            cpu_res['configuration'] |= { 'cores.enabled': int(cpu_info['configuration'][c]) }
                        elif c == 'threads':
                            cpu_res['configuration'] |= { c: int(cpu_info['configuration'][c]) }
                        else:
                            cpu_res['configuration'] |= { c: cpu_info['configuration'][c] }

            res |= { re.sub(':', '', cpu): cpu_res }

    return res

def get_memory_info(**kwargs):
    res = {}
    if 'lshw' in _components_info:
        memories = lshw_get_ids(_components_info['lshw']['content'], '^memory')
        for mem in memories:
            memory = lshw_get_desc(_components_info['lshw']['content'], mem)
            if 'size' not in memory:
                # memory controller w/o description we need
                continue

            def pretty_size(size, units):
                if units != 'bytes':
                    return { 'value': size, 'units': units }
                return { 'value': int(bytes_to(size, 'GiB')), 'units': 'GiB' }

            # Filling in info we would get running lshw from non-privileged user
            mem_res = {
                'name': 'System Memory',
                'vendor': 'Unknown',
                'size': pretty_size(memory['size'], memory['units'])
            }

            # Going into details if lshw was run from privileged user
            if 'children' in memory:
                slots_total = 0
                slots_taken = 0
                slots_map = []
                same_mem = True
                slot = {}
                for child in memory['children']:
                    slots_total += 1
                    if child['product'] == 'NO DIMM':
                        slots_map += [0]
                        continue
                    slots_taken += 1
                    slots_map += [1]
                    if not slot:
                        slot = {
                            'name': child['product'],
                            'vendor': child['vendor'],
                            'size': pretty_size(child['size'], child['units']),
                            'description': child['description']
                        }
                        continue
                    if child['product'] != slot['name'] or child['vendor'] != child['vendor']:
                        same_mem = False

                mem_res |= {
                    'capacity': pretty_size(memory['capacity'], memory['units']),
                    'slots.total': slots_total,
                    'slots.taken': slots_taken,
                    'slots.map': slots_map
                }
                if 'configuration' in memory:
                    mem_res |= { 'configuration': memory['configuration']}
                if same_mem:
                    freq_match = re.match('.*( [0-9]+ MHz) .*', slot['description'])
                    freq = freq_match.group(1) if freq_match.group(1) is not None else ''
                    mem_res |= {
                        'name': slot['name'],
                        'vendor': slot['vendor'],
                        'description': str(slots_taken) + 'x' + str(slot['size']['value']) + slot['size']['units'] + freq
                    }
                else:
                    mem_res |= { 'name': 'Multiple names', 'vendor': 'Multiple vendors' }

            res |= { re.sub(':', '', mem): mem_res }

    return res

def get_hardware(**kwargs):
    global _components_info

    res = {
        'product': {
            'name': read_sysfs_file('/sys/devices/virtual/dmi/id/product_name', **kwargs),
            'version': read_sysfs_file('/sys/devices/virtual/dmi/id/product_version', **kwargs),
        },
        'board': {
            'name': read_sysfs_file('/sys/devices/virtual/dmi/id/board_name', **kwargs),
            'vendor': read_sysfs_file('/sys/devices/virtual/dmi/id/board_vendor', **kwargs),
            'version': read_sysfs_file('/sys/devices/virtual/dmi/id/board_version', **kwargs),
            'bios': {
                'date': read_sysfs_file('/sys/devices/virtual/dmi/id/bios_date', **kwargs),
                'vendor': read_sysfs_file('/sys/devices/virtual/dmi/id/bios_vendor', **kwargs),
                'version': read_sysfs_file('/sys/devices/virtual/dmi/id/bios_version', **kwargs)
            }
        }
    }

    drm = os.listdir('/sys/class/drm/')
    drm.sort()
    for card in drm:
        if re.match('card[0-9]*$', card):
            res |= { re.sub('card', 'gpu', card): get_gpu_info('/sys/class/drm/' + card, **kwargs) }

    res |= get_cpu_info(**kwargs)
    res |= get_memory_info(**kwargs)

    return res

def get_system_config(**kwargs):
    res = {}
    if is_docker():
        res = {
            'docker0': {
                'baremetal': 'baremetal0',
                'docker': {}, # a placeholder to amend later
                'software': get_software(**kwargs)
            }
        }
    else:
        # assuming baremetal
        res = {
            'baremetal0': {
                'software': get_software(**kwargs),
                'hardware': get_hardware(**kwargs)
            }
        }
    return res

def get_sysinfo(**kwargs):
    res = {}
    files = {}

    # 'schema' points to json-schema output is compliant to
    # TBD for now, need to replace with URL of the schema
    res |= { 'schema': 'TBD' }
    # adding empty 'config' here to enforce it to be dumped before 'files'
    res |= { 'config': {} }

    if enabled('cuda', **kwargs):
        files |= get_cuda_info(**kwargs)

    if enabled('svrinfo', **kwargs):
        files |= get_svrinfo(**kwargs)

    if enabled('docker', **kwargs):
        files |= get_docker_info(**kwargs)

    if enabled('dpkg', **kwargs):
        files |= get_dpkg_info(**kwargs)

    if enabled('lshw', **kwargs):
        files |= get_lshw_info(**kwargs)

    if files:
        res['files'] = files

    res |= { 'config': { 'system': get_system_config(**kwargs) } }

    return res

def get_parser():
    parser = argparse.ArgumentParser(
        prog='js_sysinfo',
          description='Dump system information according to JSON schema',
          epilog='Copyright (c) 2024 Intel Corporation')

    parser.add_argument('-v', '--verbose', action="store_true", help='Enable verbose output')
    parser.add_argument('-q', '--quiet', action="store_true", help='Be quiet and suppress messages to stdout')

    group = parser.add_argument_group('collect information options')
    group.add_argument('-a', '--all', action="store_true", help='collect information for all known components')
    group.add_argument('--verify', action=argparse.BooleanOptionalAction, default=False,
        help='verify integrity of ALL the packages (select packages are always verified)')
    # using 'store_const' to support triplet options with values [None, True, False]
    group.add_argument('--cuda', action="store_const", const=True, help='collect information for CUDA')
    group.add_argument('--dkms', action="store_const", const=True, help='collect information for DKMS')
    group.add_argument('--docker', action="store_const", const=True, help='collect information for Docker')
    group.add_argument('--dpkg', action="store_const", const=True, help='collect information for some key DPKG packages')
    group.add_argument('--lshw', action="store_const", const=True, help='collect information with lshw')
    group.add_argument('--pytorch', action="store_const", const=True, help='collect information for Pytorch')
    group.add_argument('--svrinfo', action="store_const", const=True, help='collect information with svr-info (should be in PATH)')
    group.add_argument('--xpu', action="store_const", const=True, help='collect information for xpu')

    group1 = parser.add_argument_group('JSON dump options')
    group1.add_argument('-o', '--output', action="store", type=str, default='', help='File to store output')
    group1.add_argument('--indent', default=None, help='indent for json.dump()')

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()

    indent = int(args.indent) if args.indent and args.indent.isdecimal() else args.indent
    sysinfo = get_sysinfo(**vars(args))

    if args.output == '':
        json.dump(sysinfo, sys.stdout, indent=indent)
    else:
        with open(args.output, 'w') as f:
            json.dump(sysinfo, f, indent=indent)
