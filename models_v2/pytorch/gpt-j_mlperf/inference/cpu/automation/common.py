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
import logging
import subprocess
import sys

# Logger 
log_file = 'mlperf.log'
file_handler = logging.FileHandler(log_file)
stdout_handler = logging.StreamHandler(stream=sys.stdout)
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s][%(levelname)s] %(filename)s:%(lineno)-5d - %(message)s',
    handlers=[stdout_handler, file_handler]
)
logger = logging.getLogger()

def run_bash_with_live_output(cmd: str, is_container_log: bool, env=None):
    logger.info(f'Executing bash commands: {cmd}')
    import shlex
    cmds = shlex.split('/bin/sh -c')
    cmds.append(cmd)
    if env:
        process = subprocess.Popen(cmds, stdout=subprocess.PIPE, env=env)
    else:
        process = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    while True:
        # stdout, stderr = process.communicate()
        output = process.stdout.readline() if process.stdout else ''
        error = process.stderr.readline() if process.stderr else ''
        # output = stdout.readline() if stdout else ''
        # error = stderr.readline() if stderr else ''
        if (output == '' or error == '') and process.poll() is not None:
            break
        if output:
            output = output.decode() if type(output) == bytes else output
            if is_container_log:
                logger.debug(output.strip())
            else:
                logger.info(output.strip())
        if error:
            error = error.decode() if type(error) == bytes else error
            if is_container_log:
                logger.debug(error.strip())
            else:
                logger.error(error.strip())
    rc = process.poll()
    return rc
 
def run_bash(cmd: str, silent: bool=False, ignore_err: bool=False):
    if not silent:
        logger.info(f'Executing bash commands: {cmd}')
    import shlex
    cmds = shlex.split('/bin/sh -c')
    cmds.append(cmd)
    process = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    output, error = process.communicate()
    stdout = output.decode('utf-8') if output else ""
    stderr = error.decode('utf-8') if error else ""
    # ret = subprocess.run(cmd, stdout=subprocess.PIPE)
    # stdout = ret.stdout.decode()
    # stderr = ret.stderr.decode()
    if not silent:
        logger.info(f'stdout: {stdout}')
        logger.info(f'stderr: {stderr}')
    # if ret.returncode != 0 and not ignore_err:
    if process.returncode != 0 and not ignore_err:
        raise RuntimeError(f'Failed to execute command, stderr={stderr}')
    else:
        if not silent:
            logger.info(f'Successfully executed bash commands.')
        return stdout, stderr

def check_fields_in_file(file_path, field1, field2):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if field1 in line and field2 in line:
                    return True
        return False
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return False
