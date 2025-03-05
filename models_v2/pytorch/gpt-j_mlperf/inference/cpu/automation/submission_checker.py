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
import time
import argparse
from common import *
import logging
import shutil
from conf import RuntimeConfig

log_file = 'mlperf_submission_checker.log'
if os.path.exists(log_file):
    os.remove(log_file)
file_handler = logging.FileHandler(log_file)
stdout_handler = logging.StreamHandler(stream=sys.stdout)
logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(levelname)s] %(filename)s:%(lineno)-4d - %(message)s',
    handlers=[stdout_handler, file_handler]
)
logger = logging.getLogger()

LOCAL_AUTOMATION_DIR = os.getcwd()
f_config = os.path.join(LOCAL_AUTOMATION_DIR, 'config.yaml')
config = RuntimeConfig(f_config, logger)
ORG = config.get_org()
VERSION = config.get_version()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", 
        help="Input directory. The output directory of run.py",
        type=str, 
        required=True,
    )
    parser.add_argument(
        "-o", "--output-dir", 
        help="Ouput directory",
        type=str, 
        required=True,
    )
    parser.add_argument(
        "-r", "--runtime-dir", 
        help="Runtime directory",
        default="/tmp/mlperf/runtime",
        type=str
    )
    args = parser.parse_args()
    return args

def check_args(args):
    assert os.path.exists(args.input_dir), f'Path not found: {args.input_dir}'

    if os.path.exists(args.output_dir):
        logger.warning(f'Output dir already existed: {args.output_dir}')
        ts = int(time.time())
        backup_dir = f'{args.output_dir}_{ts}'
        shutil.move(args.output_dir, backup_dir)
        logger.info(f'Move the output dir to {backup_dir}')

    if os.path.exists(args.runtime_dir):
        shutil.rmtree(args.runtime_dir)
    os.makedirs(args.runtime_dir)
    
def fetch_mlperf_infer(args):
    logger.info('Fetching MLPerf inference kit...')
    run_bash(
        f"cd {args.runtime_dir}; " \
        f"git clone --recurse-submodules https://github.com/mlcommons/inference.git")

def truncate_logs(args):
    logger.info('-' * 40 )
    logger.info('Truncating accuracy logs...')
    logger.info('-' * 40 )
    cmd = f"cd {args.runtime_dir}/inference; " \
          f"python3 ./tools/submission/truncate_accuracy_log.py " \
                f"--input {args.input_dir} " \
                f"--submitter {ORG} " \
                f"--output {args.output_dir}"
    run_bash(cmd)

def is_historical_dir(path):
    return os.path.isdir(path) and (
           'run_1_' in path or \
           'accuracy_' in path or \
           'Offline_' in path or \
           'Server_' in path)

def remove_logs(dir):
    if os.path.isdir(dir):
        for f in os.listdir(dir):
            p = os.path.join(dir, f)
            if os.path.isdir(p):
                if is_historical_dir(p):
                    logger.debug(f'Removed {f}.')
                    shutil.rmtree(p)
                else:
                    remove_logs(p)

def clear_historical_logs(args):
    logger.info('Clearing historical logs...')
    remove_logs(args.output_dir)
    
def do_submission_checker(args):
    logger.info('-' * 40 )
    logger.info('Performing submission checker...')
    logger.info('-' * 40 )
    cmd = f"cd {args.runtime_dir}/inference; " \
          f"python3 ./tools/submission/submission_checker.py " \
                f"--input {args.output_dir} " \
                f"--submitter {ORG} " \
                f"--version v{VERSION}"
    run_bash(cmd)

args = parse_args()
check_args(args)
fetch_mlperf_infer(args)
truncate_logs(args)
clear_historical_logs(args)
do_submission_checker(args)
