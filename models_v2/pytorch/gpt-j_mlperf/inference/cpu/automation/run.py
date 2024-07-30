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
import os
import sys
import argparse
import logging
from conf import RuntimeConfig
from tabulate import tabulate
from python_on_whales import docker

log_file = 'mlperf.log'
if os.path.exists(log_file):
    os.remove(log_file)
file_handler = logging.FileHandler(log_file)
stdout_handler = logging.StreamHandler(stream=sys.stdout)
logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s][%(levelname)s] %(filename)s:%(lineno)-4d - %(message)s',
    handlers=[stdout_handler, file_handler]
)
logger = logging.getLogger()

LOCAL_AUTOMATION_DIR = os.getcwd()
f_config = os.path.join(LOCAL_AUTOMATION_DIR, 'config.yaml')
logger.info(f'Loading configurations from {f_config}.')
config = RuntimeConfig(f_config, logger)

SUPPORTED_MODELS = config.get_model_names()
SUPPORTED_DTYPES = ['int8', 'bf16', 'fp16', 'int4', 'fp32', 'mix']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--model", 
        help="Benchmarking model",
        choices=SUPPORTED_MODELS, 
        default=None, 
        required=True,
    )
    parser.add_argument(
        "-i", "--implementation", 
        help="Implementation id",
        default="pytorch-cpu",
        type=str, 
    )
    parser.add_argument(
        "-y", "--dtype", 
        help="Precision",
        choices=SUPPORTED_DTYPES, 
        default=SUPPORTED_DTYPES[0]
    )
    parser.add_argument(
        "-d", "--dataset-dir", 
        help="path of the datasets",
        default=None, 
        required=True,
        type=str
    )
    parser.add_argument(
        "-m", "--model-dir", 
        help="path of the models", 
        default=None, 
        required=True,
        type=str
    )
    parser.add_argument(
        "-t", "--output", 
        help="path of the outputs",
        default=None, 
        required=True,
        type=str
    )
    parser.add_argument(
        "-x", "--container-name-suffix",
        help="The suffix of docker container name, used for avoiding name conflicts.", 
        default=None, 
        required=True,
        type=str
    )
    parser.add_argument(
        "-p", "--performance-only", 
        help="The option of running performance test only.",
        action="store_true", 
        required=False
    )
    parser.add_argument(
        "-a", "--accuracy-only", 
        help="The option of running accuracy test only.",
        action="store_true", 
        required=False
    )
    parser.add_argument(
        "-o", "--offline-only", 
        help="The option of running offline scenario only.",
        action="store_true", 
        required=False
    )
    parser.add_argument(
        "-s", "--server-only", 
        help="The option of running server scenario only.",
        action="store_true", 
        required=False
    )
    parser.add_argument(
        "-c", "--compliance-only", 
        help="The option of running compliance test only.",
        action="store_true", 
        required=False
    )    
    parser.add_argument(
        "-b", "--skip-docker-build", 
        help="The option of skipping building docker image.",
        action="store_true", 
        required=False
    )
    parser.add_argument(
        "-u", "--skip-create-container", 
        help="The option of skipping docker build and container creation.",
        action="store_true", 
        required=False
    )      
    parser.add_argument(
        "-r", "--skip-data-preprocess", 
        help="The option of skipping data preprocessing.",
        action="store_true", 
        required=False
    )  
    parser.add_argument(
        "-z", "--ci-run",
        help="option of running CI runs",
        action="store_true",
        required=False
    )
    args = parser.parse_args()
    return args


def check_args(args):
    assert os.path.exists(args.dataset_dir), f'Path not found: {args.dataset_dir}'
    assert os.path.exists(args.model_dir), f'Path not found: {args.model_dir}'
    assert os.path.exists(args.output), f'Path not found: {args.output}'

logger.info('Parsing args:')
args = parse_args()
check_args(args)
logger.info(f'args: {args}')

MODEL = args.model
IMPL = args.implementation
DTYPE = args.dtype
BASE_DIR = os.path.abspath(
    os.path.join(
        LOCAL_AUTOMATION_DIR, 
        os.pardir, os.pardir, os.pardir, os.pardir
    )
)
#LOCAL_CODE_DIR = os.path.join(
#    BASE_DIR, config.get_division(MODEL), 
#    config.get_org(), 'code', MODEL, IMPL
#)
LOCAL_CODE_DIR = os.path.join(
    BASE_DIR, 'gpt-j_mlperf', 'inference', 'cpu', MODEL, IMPL
)
LOCAL_DATA_DIR = args.dataset_dir
LOCAL_MODEL_DIR = args.model_dir
LOCAL_OUTPUT_DIR = args.output
CONTAINER_AUTOMATION_DIR = config.get_container_automation_dir(MODEL, IMPL, DTYPE)
CONTAINER_CODE_DIR = config.get_container_workdir(MODEL, IMPL, DTYPE)
CONTAINER_DATA_DIR = config.get_container_data_dir(MODEL, IMPL, DTYPE)
CONTAINER_MODEL_DIR = config.get_container_model_dir(MODEL, IMPL, DTYPE)
CONTAINER_OUTPUT_DIR = config.get_container_output_dir(MODEL, IMPL, DTYPE)

PREPARE = True
PERFORMANCE = True
ACCURACY = True
COMPLIANCE = True
OFFLINE = True
SERVER = True
SENSORS = True
CIRUN=False

if args.performance_only:
    ACCURACY = False
    COMPLIANCE = False

if args.accuracy_only:
    PERFORMANCE = False
    COMPLIANCE = False

if args.offline_only:
    SERVER = False
    # COMPLIANCE = False

if args.server_only:
    OFFLINE = False
    # COMPLIANCE = False

if args.compliance_only:
    PERFORMANCE = False
    ACCURACY = False

if args.skip_data_preprocess:
    PREPARE = False

if args.ci_run:
    CIRUN = True

def check_docker_args(config: RuntimeConfig):
    docker_image_id = config.get_docker_image_name(MODEL, IMPL, DTYPE)
    docker_image_version = config.get_version()
    docker_builder = config.get_image_builder(MODEL, IMPL, DTYPE)
    assert docker_image_id, f'docker_image_id is not found in config.yaml.'
    assert docker_image_version, f'docker_image_version is not found in config.yaml.'
    assert docker_builder, f'docker_builder is not found in config.yaml.'

def docker_build():  
    logger.info(f'Building docker image for {MODEL}/{IMPL}/{DTYPE}.')
    check_docker_args(config)
    docker_image_id = config.get_docker_image_name(MODEL, IMPL, DTYPE)
    docker_image_version = config.get_version()
    docker_builder = config.get_image_builder(MODEL, IMPL, DTYPE)

    buildargs = {}
    for arg in ['ftp_proxy', 'http_proxy', 'https_proxy', 'no_proxy', 'socks_proxy']:
        try:
            buildargs[arg] = os.environ[arg]
        except KeyError:
            pass
        try:
            buildargs[arg.upper()] = os.environ[arg.upper()]
        except KeyError:
            pass
    buildargs['BENCHMARK'] = MODEL
    buildargs['IMPL'] = IMPL

    image = docker.build(
        context_path = f'{LOCAL_CODE_DIR}/../../../',
        file = f'{LOCAL_CODE_DIR}/docker/{docker_builder}',
        build_args = buildargs,
        cache = False,
        tags = f'{docker_image_id}:{docker_image_version}'
    )
    assert image, f'Failed to build docker image: {image.id}'
    logger.info(f'Successfully built image: {image.id}')
    return image

def docker_run(image):
    logger.info(f'Launching docker container for {MODEL}.')
    container = docker.run(
        image = image,
        name = f'intel_{MODEL}_{DTYPE}_{args.container_name_suffix}',
        privileged = True,
        networks = ['host'],
        ipc = 'host',
        tty = True,
        interactive = True,
        detach = True,
        volumes = [(LOCAL_AUTOMATION_DIR, CONTAINER_AUTOMATION_DIR),
                   (LOCAL_DATA_DIR, CONTAINER_DATA_DIR),
                   (LOCAL_MODEL_DIR, CONTAINER_MODEL_DIR),
                   (LOCAL_OUTPUT_DIR, CONTAINER_OUTPUT_DIR)])
    assert container.state.running, f'Failed to launch container for image: {image.id}.'
    logger.info(f'Successfully launched container: {container.id[0:12]} for image: {image.id[0:19]}')
    return container

print('Path mapping:')
print(tabulate([['code dir', LOCAL_CODE_DIR, CONTAINER_CODE_DIR], 
                ['automation kit dir', LOCAL_AUTOMATION_DIR, CONTAINER_AUTOMATION_DIR],
                ['data dir', LOCAL_DATA_DIR, CONTAINER_DATA_DIR],
                ['model dir', LOCAL_MODEL_DIR, CONTAINER_MODEL_DIR],
                ['output dir', LOCAL_OUTPUT_DIR, CONTAINER_OUTPUT_DIR]]
                , headers=[' ', 'Local', 'Container'], tablefmt='fancy_grid'
                , maxcolwidths=[None, 40, 40]))

print('Runtime options:')
print(tabulate([['prepare', PREPARE], 
                ['performance', PERFORMANCE],
                ['accuracy', ACCURACY],
                ['compliance', COMPLIANCE],
                ['offline', OFFLINE],
                ['server', SERVER],
                ['sensors', SENSORS]]
                , headers=['Option', 'Value'], tablefmt='fancy_grid'))

if SENSORS:
    logger.info('Collecting environment information on baremetal.')
    from common import *
    run_bash(
        f"cd {LOCAL_AUTOMATION_DIR}; " \
        f"is_in_container=False " \
        f"base_path={BASE_DIR} output_path={LOCAL_OUTPUT_DIR} " \
        f"model={MODEL} impl={IMPL} dtype={DTYPE} " \
        "bash get_env_info.sh", 
        True
    )

if not args.skip_create_container:
    if args.skip_docker_build:
        logger.info('User skipped docker build.')
        image_id = input('>>> Enter your docker image id:')
        image = docker.image.inspect(image_id)
    else:
        image = docker_build()
    logger.info(f'{"="*5} Using docker image: {image.id[0:19]} {"="*5}')

if args.skip_create_container:
    logger.info('User skipped docker build and container creation.')
    container_id = input('>>> Enter your container id:')
    container = docker.container.inspect(container_id)
else:
    container = docker_run(image)

logger.info(f'{"="*5} Running workloads in container: {container.id[0:12]} {"="*5}')

cmd_options = ''
cmd_options += ' -r' if PREPARE else ''
cmd_options += ' -p' if PERFORMANCE else ''
cmd_options += ' -a' if ACCURACY else ''
cmd_options += ' -c' if COMPLIANCE else ''
cmd_options += ' -e' if SENSORS else ''
cmd_options += ' -o' if OFFLINE else ''
cmd_options += ' -s' if SERVER else ''
cmd_options += ' -z' if CIRUN else ''

proxy_cmd = ''
try:
    HTTP_PROXY = os.environ['http_proxy']
    proxy_cmd = f'{proxy_cmd} export http_proxy={HTTP_PROXY};'
except KeyError:
    pass
try:
    HTTPS_PROXY = os.environ['https_proxy']
    proxy_cmd = f'{proxy_cmd} export https_proxy={HTTPS_PROXY};'
except KeyError:
    pass    

docker_exec_cmd = f'{proxy_cmd}' \
                  f'cd {CONTAINER_AUTOMATION_DIR};' \
                  f'python benchmark.py {cmd_options} ' \
                  f'-m {MODEL} -i {IMPL} -y {DTYPE}'
output_generator = docker.execute(container, command=['sh', '-c', docker_exec_cmd], stream=True)
for stream_type, stream_content in output_generator:
    logger.info(stream_content.decode())
