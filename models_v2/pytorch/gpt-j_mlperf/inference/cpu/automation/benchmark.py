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
import time
import argparse
import shutil
import logging
from conf import RuntimeConfig
from common import *

f_config = 'config.yaml'
print(f'Loading configurations from {f_config}.')
config = RuntimeConfig(f_config, logging.getLogger())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        help="model name", 
        default=None, 
        required=True, 
        type=str
    )
    parser.add_argument(
        "-i", "--implementation",
        help="implementation id", 
        default=None, 
        required=True, 
        type=str
    )
    parser.add_argument(
        "-y", "--dtype", 
        help="Precision",
        default=None, 
        required=True,
        type=str
    )
    parser.add_argument(
        "-r", "--prepare", 
        help="option of data preprocessing",
        action="store_true",
        required=False
    )
    parser.add_argument(
        "-p", "--performance",
        help="option of running performance test only",
        action="store_true", 
        required=False
    )
    parser.add_argument(
        "-a", "--accuracy",
        help="option of running accuracy test only", 
        action="store_true", 
        required=False
    )
    parser.add_argument(
        "-c", "--compliance",
        help="option of running compliance test only",
        action="store_true", 
        required=False
    )
    parser.add_argument(
        "-o", "--offline", 
        help="option of running offline scenario only",
        action="store_true", 
        required=False,
    )
    parser.add_argument(
        "-s", "--server", 
        help="option of running server scenario only",
        action="store_true",
        required=False
    )
    parser.add_argument(
        "-e", "--sensors", 
        help="option of collecting environment information",
        action="store_true", 
        required=False
    )
    parser.add_argument(
        "-z", "--cirun",
        help="option of running CI runs",
        action="store_true",
        required=False
    )
    args = parser.parse_args()
    return args

print('Parsing args:')
args = parse_args()
print(f'args in container: {args}')

model = args.model
impl = args.implementation
dtype = args.dtype
RUN_INTERVAL = config.get_run_interval()
CONTAINER_DATA_DIR = config.get_container_data_dir(model, impl, dtype)
CONTAINER_CODE_DIR = config.get_container_workdir(model, impl, dtype)
CONTAINER_OUTPUT_DIR = config.get_container_output_dir(model, impl, dtype)
CONTAINER_MODEL_DIR = config.get_container_model_dir(model, impl, dtype)
CONTAINER_AUTOMATION_DIR = config.get_container_automation_dir(model, impl, dtype)
CONTAINER_COMPLIANCE_SUITE_DIR = config.get_container_compliance_suite_dir(model, impl, dtype)
TIMESTAMP = int(time.time())
global EXPORT_ENV_PARAMS

class TestResult:
    results = {}

    def __init__(self, ts, output_dir):
        self.ts = ts
        self.output_dir = output_dir

    def get_key(self, model, impl, dtype, scenario):
        return f'{model}/{impl}/{dtype}/{scenario}'

    def create_dict_if_not_exists(self, model, impl, dtype, scenario):
        key = self.get_key(model, impl, dtype, scenario)
        if not self.results.get(key):
            self.results[key] = {}
            rs = self.results.get(key)
            rs['timestamp'] = self.ts
            rs['model'] = model
            rs['implementation'] = impl
            rs['dtype'] = dtype
            rs['scenario'] = scenario
            rs['qps'] = 'n/a'
            rs['percentile_99%'] = 'n/a'
            rs['accuracy'] = 'n/a'
            rs['TEST01'] = 'n/a'
            rs['TEST04'] = 'n/a'
            rs['TEST05'] = 'n/a'

    def add_accuracy_result(self, model, impl, dtype, scenario, acc):
        self.create_dict_if_not_exists(model, impl, dtype, scenario)
        self.results.get(self.get_key(model, impl, dtype, scenario))['accuracy'] = acc

    def add_performance_result(self, model, impl, dtype, scenario, qps, percentile_99):
        self.create_dict_if_not_exists(model, impl, dtype, scenario)
        self.results.get(self.get_key(model, impl, dtype, scenario))['qps'] = qps
        self.results.get(self.get_key(model, impl, dtype, scenario))['percentile_99%'] = percentile_99

    def add_compliance_test_results(self, model, impl, dtype, scenario, test, result):
        self.create_dict_if_not_exists(model, impl, dtype, scenario)
        self.results.get(self.get_key(model, impl, dtype, scenario))[test] = result

    def flush(self):
        f_path = os.path.join(self.output_dir, 'results.csv')
        with open(f_path, 'a') as f:
            cols = ['timestamp', 'model', 'implementation', 
                    'dtype', 'scenario', 'qps', 'percentile_99%', 
                    'accuracy', 'TEST01', 'TEST04', 'TEST05']
            import csv
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writerow({})
            writer.writeheader()
            for row in self.results.values():
                writer.writerow(row)

result = TestResult(TIMESTAMP, CONTAINER_OUTPUT_DIR)

def export_env(model: str, impl: str, dtype: str):
    global EXPORT_ENV_PARAMS
    print('Exporting environment variables.')
    if args.cirun:
        ci_run=1
    else:
        ci_run=0
    EXPORT_ENV_PARAMS = f'export CONTAINER_MODEL_DIR={CONTAINER_MODEL_DIR} && ' \
                        f'export CONTAINER_DATA_DIR={CONTAINER_DATA_DIR} && ' \
                        f'export CONTAINER_CODE_DIR={CONTAINER_CODE_DIR} && ' \
                        f'export CONTAINER_OUTPUT_DIR={CONTAINER_OUTPUT_DIR} && ' \
                        f'export CONTAINER_COMPLIANCE_SUITE_DIR={CONTAINER_COMPLIANCE_SUITE_DIR} && ' \
                        f'export MODEL={model} && ' \
                        f'export IMPL={impl} && ' \
                        f'export DTYPE={dtype} && ' \
                        f'export ci_run={ci_run}'
    MODEL_PARAMS = config.get_params(model, impl, dtype)
    if MODEL_PARAMS:
        EXPORT_ENV_PARAMS = f'{EXPORT_ENV_PARAMS} && {MODEL_PARAMS} '

def prepare(model: str, impl: str, dtype: str):
    global EXPORT_ENV_PARAMS
    if args.prepare:
        print(f'{"="*5} Preprocessing for {model} {"="*5}')
        start_ts = time.time()
        mlperf_inference_dir = f'{config.get_container_compliance_suite_dir(model, impl, dtype)}/../../'
        mlperf_conf_path = os.path.join(mlperf_inference_dir, 'mlperf.conf')
        shutil.copy2(mlperf_conf_path, CONTAINER_CODE_DIR)

        cmd = config.get_preproc_cmds(model, impl, dtype)
        rc = run_bash_with_live_output(f'{EXPORT_ENV_PARAMS} && {cmd}', True)
        assert rc == 0, f'Failed to preprocess data and model: {model}, container log: ' \
                        f'{CONTAINER_OUTPUT_DIR}/preproc_{model}_{impl}_{dtype}.log'
        print(f"Elapsed time: {time.time() - start_ts} sec.")
        print(f"Completed preprocessing for model {model}.")
    else:
        print('User skipped preprocessing.')

def collect_env_info():
    if args.sensors:
        print('Collecting environment information in container.')
        run_bash(
            f'cd {CONTAINER_AUTOMATION_DIR} && ' \
            f"is_in_container=True " \
            f"base_path='' output_path={CONTAINER_OUTPUT_DIR} " \
            f"model={model} impl={impl} dtype={dtype} " \
            f'bash get_env_info.sh', 
            True
        )

def isOffline(scenario: str):
    return scenario == 'Offline' and args.offline

def isServer(scenario: str): 
    return scenario == 'Server' and args.server

def is_run_scenario(model: str, impl: str, dtype: str, scenario: str):
    return (isOffline(scenario) or isServer(scenario)) and \
            config.check_scenario_support(model, impl, dtype, scenario)

def isAccuracy(mode: str):
    return mode == 'accuracy' and args.accuracy

def isPerformance(mode: str):
    return mode == 'performance' and args.performance

def isCompliance(mode: str):
    return mode == 'compliance' and args.compliance

def isModeSupported(model: str, impl: str, dtype: str, scenario: str, mode: str):
    return config.check_test_mode_support(model, impl, dtype, scenario, mode) \
           or mode == 'compliance'

def is_run_mode(model: str, impl: str, dtype: str, scenario: str, mode: str):
    return (isAccuracy(mode) or isPerformance(mode) or isCompliance(mode)) \
           and isModeSupported(model, impl, dtype, scenario, mode)

def post_process_run_output(model: str, impl: str, dtype: str, scenario: str, mode: str):
    TEST_LOG_DIR = os.path.join(
        CONTAINER_CODE_DIR, 
        config.get_container_log_dir(model, impl, dtype, scenario, mode)
    )   
    assert os.path.exists(TEST_LOG_DIR), f'Benchmark log dir not found: {TEST_LOG_DIR}.'

    OUTPUT_RESULT_DIR = os.path.join(
        config.get_output_result_dir(model, impl, dtype), 
        scenario, mode
    )
    if mode == 'performance':
        OUTPUT_RESULT_DIR = os.path.join(OUTPUT_RESULT_DIR, 'run_1')
    if os.path.exists(OUTPUT_RESULT_DIR):
        # backup results
        shutil.move(OUTPUT_RESULT_DIR, f'{OUTPUT_RESULT_DIR}_{TIMESTAMP}')
    os.makedirs(OUTPUT_RESULT_DIR)
    
    # Copy result files from log dir to output dir
    result_files = {
        "mlperf_log_accuracy.json", 
        "mlperf_log_detail.txt", 
        "mlperf_log_summary.txt", 
        "accuracy.txt"
    }

    for f in result_files:
        if mode != 'accuracy' and f == 'accuracy.txt':
            continue
        f_result = os.path.join(TEST_LOG_DIR, f) 
        if os.path.exists(f_result):
           shutil.copy2(f_result, OUTPUT_RESULT_DIR)

    if mode == 'performance':
        # Assert if performance run is valid
        cmd = f'cat {TEST_LOG_DIR}/mlperf_log_summary.txt |awk ' + '\'{print $4}\' |grep -x VALID' 
        stdout, _ = run_bash(cmd, silent=True, ignore_err=True)
        if not 'VALID' in stdout:
            print(f'!!WARN!! INVALID performance: {model}/{impl}/{dtype}/{scenario}/{mode}')

        # Print performance
        stdout, _ = run_bash(f'cat {OUTPUT_RESULT_DIR}/mlperf_log_summary.txt | grep target_qps', True)
        target_qps = float(stdout.split(' ')[-1])

        stdout, _ = run_bash(f'cat {OUTPUT_RESULT_DIR}/mlperf_log_summary.txt | grep "per second"', True)
        output_qps = float(stdout.split("\n")[0].split(' ')[-1])

        stdout, _ = run_bash(f'cat {OUTPUT_RESULT_DIR}/mlperf_log_summary.txt | grep "99.00 percentile latency"', True)
        latency = float(stdout.split(' ')[-1])

        result.add_performance_result(model, impl, dtype, scenario, output_qps, latency)

        print('*' * 80)
        print(f'{model}/{impl}/{dtype}/{scenario}/{mode}:')
        print(f'  Target QPS: {target_qps}')
        print(f'  Perf QPS: {output_qps}')
        print(f'  99.00 percentile latency: {latency}')
        print(f'  Result dir: {OUTPUT_RESULT_DIR}')
        print('*' * 80)
    
    if mode == 'accuracy':
        # print accuracy
        cmd = f'cat {TEST_LOG_DIR}/accuracy.txt'
        stdout, _ = run_bash(cmd, True)
        result.add_accuracy_result(model, impl, dtype, scenario, stdout.strip().replace('\n', ''))

        print('*' * 80)
        print(f'{model}/{impl}/{dtype}/{scenario}/{mode}:')
        print(f'  Accuracy: {stdout}')
        print(f'  Result dir: {OUTPUT_RESULT_DIR}')
        print('*' * 80)
        
def clear_cache():
    print('Clearing cache and setting environment parameters...')
    run_bash(f'cd {CONTAINER_CODE_DIR} && bash run_clean.sh', True)

def run_model(model: str, impl: str, dtype: str, scenario: str, mode: str):
    global EXPORT_ENV_PARAMS
    print(f'Sleep {RUN_INTERVAL} seconds.')
    time.sleep(int(RUN_INTERVAL))
    clear_cache()
    cmd = config.get_exec_cmd(model, impl, dtype, scenario, mode)
    cmd = f'{EXPORT_ENV_PARAMS} && cd {CONTAINER_CODE_DIR} && {cmd}'
    run_bash_with_live_output(cmd, True)


def save_measurements(model: str, impl: str, dtype: str, scenario: str, mode: str):
    print(f'Updating measurements of {model}/{impl}/{dtype}/{scenario}/{mode}.')
    MEASUREMENT_DIR = config.get_output_measurements_dir(model, impl, dtype, scenario)

    # Save mlperf.conf and user.conf/user_*.conf
    for f in ['mlperf.conf', 'user.conf', f'user_{dtype}.conf']:
        src = os.path.join(
            CONTAINER_CODE_DIR, 
            config.get_model_conf_path(model, impl, dtype), 
            f
        )
        if os.path.exists(src):
            shutil.copy2(src, MEASUREMENT_DIR)
    
    # Save READ.MD
    shutil.copy2(os.path.join(CONTAINER_CODE_DIR, 'DEV.md'), MEASUREMENT_DIR)
        
    # Save bootstrap script
    f = os.path.join(
        CONTAINER_CODE_DIR, 
        config.get_script(model, impl, dtype, scenario, mode)
    )
    shutil.copy2(f, MEASUREMENT_DIR)

def run_mode(model: str, impl: str, dtype: str, scenario: str, mode: str):
    print(f'{"="*5} Performing {model}/{impl}/{dtype}/{scenario}/{mode} {"="*5}')
    try:
        run_model(model, impl, dtype, scenario, mode)
    except Exception as e:
        print(e)
        if mode =='accuracy':
            result.add_accuracy_result(model, impl, dtype, scenario, 'INVALID_OR_ERROR')
        elif mode == 'performance':
            result.add_performance_result(model, impl, dtype, scenario, 'INVALID_OR_ERROR', 'INVALID_OR_ERROR')
        return
    post_process_run_output(model, impl, dtype, scenario, mode)
    save_measurements(model, impl, dtype, scenario, mode)

def get_compliance_dir(suite_dir: str, test_id: str, model: str, impl: str, dtype: str):
    return os.path.join(suite_dir, test_id)

def get_audit_conf_path(suite_dir: str, test_id: str, model: str, impl: str, dtype: str):
    COMPLIANCE_DIR = get_compliance_dir(suite_dir, test_id, model, impl, dtype)
    if 'TEST01' == test_id:
        model_name = config.get_compliance_test01_model_name(model, impl, dtype)
        COMPLIANCE_DIR = os.path.join(COMPLIANCE_DIR, model_name)
    return os.path.join(COMPLIANCE_DIR, 'audit.config')

def run_compliance(model: str, impl: str, dtype: str, scenario: str, mode: str):
    global EXPORT_ENV_PARAMS
    print(f'{"="*5} Performing {model}/{impl}/{dtype}/{scenario}/{mode} {"="*5}')

    COMPLIANCE_SUITE_DIR = config.get_container_compliance_suite_dir(model, impl, dtype)
    COMPLIANCE_OUTPUT_DIR = config.get_output_compliance_dir(model, impl, dtype, scenario)
    if os.path.exists(COMPLIANCE_OUTPUT_DIR):
        # backup results
        shutil.move(COMPLIANCE_OUTPUT_DIR, f'{COMPLIANCE_OUTPUT_DIR}_{TIMESTAMP}')
    OUTPUT_RESULT_DIR = os.path.join(
        config.get_output_result_dir(model, impl, dtype), scenario
    )

    compliance_tests = config.getComplianceTests(model)
    if compliance_tests is not None:
        for test in compliance_tests:
            # Copy audit config for each compliance test
            AUDIT_CONF_PATH = get_audit_conf_path(COMPLIANCE_SUITE_DIR, test, model, impl, dtype)
            shutil.copy2(AUDIT_CONF_PATH, CONTAINER_CODE_DIR)
            os.chdir(CONTAINER_CODE_DIR)
            try:
                # Perform performance run with audit.config
                print(f'Running compliance test: {model}/{impl}/{dtype}/{scenario}/{test}')
                run_model(model, impl, dtype, scenario, 'performance')
            except Exception as e:
                print(e)
                result.add_compliance_test_results(model, impl, dtype, scenario, test, 'ERROR')    
                continue
            finally:
                # Remove audit config for next run
                os.remove(os.path.join(CONTAINER_CODE_DIR, 'audit.config'))

            print(f'>>> Verifying compliance {test} result.')

            # Check if log dir exists
            COMPLIANCE_TEST_LOG_DIR = os.path.join(
                CONTAINER_CODE_DIR, 
                config.get_container_log_dir(model, impl, dtype, scenario, 'performance')
            ) 
            assert os.path.exists(COMPLIANCE_TEST_LOG_DIR), f'Benchmark log dir not found: {COMPLIANCE_TEST_LOG_DIR}.'
            
            # Check if audit info is in mlperf_log_detail.txt
            cmd = f'cat {COMPLIANCE_TEST_LOG_DIR}/mlperf_log_detail.txt|grep audit'
            stdout, _ = run_bash(cmd, True)
            assert stdout, 'Audit info not found in mlperf_log_detail.txt.'
 
            # Do compliance verfication
            COMPLIANCE_TEST_DIR = get_compliance_dir(
                COMPLIANCE_SUITE_DIR, 
                test, model, impl, dtype
            )
            cmd = f'python3 {COMPLIANCE_TEST_DIR}/run_verification.py ' \
                  f'-r {OUTPUT_RESULT_DIR} ' \
                  f'-c {COMPLIANCE_TEST_LOG_DIR} ' \
                  f'-o {COMPLIANCE_OUTPUT_DIR}'
            run_bash(cmd)

            VERIFY_PERF_PATH = os.path.join(COMPLIANCE_OUTPUT_DIR, test, 'verify_performance.txt')
            PERF_PASS = check_fields_in_file(VERIFY_PERF_PATH, 'TEST', 'PASS')

            if 'TEST01' == test:
                VERIFY_ACC_PATH = os.path.join(COMPLIANCE_OUTPUT_DIR, test, 'verify_accuracy.txt')
                ACC_PASS = check_fields_in_file(VERIFY_ACC_PATH, 'TEST', 'PASS')

                # Perform TEST01 part 3 if part2 does not pass.
                # See: https://github.com/mlcommons/inference/tree/master/compliance/nvidia/TEST01#part-iii
                if not ACC_PASS:
                    print(f'WARN: Compliance TEST01 part 2 does not pass, trying part 3.')
                    print(f'Performing compliance TEST01 part 3 step #1.')
                    ENV_PARAMS1 = f'{EXPORT_ENV_PARAMS} && ' \
                                  f'export COMPLIANCE_TEST_DIR={COMPLIANCE_TEST_DIR} && ' \
                                  f'export COMPLIANCE_OUTPUT_DIR={COMPLIANCE_OUTPUT_DIR} && ' \
                                  f'export COMPLIANCE_TEST_LOG_DIR={COMPLIANCE_TEST_LOG_DIR} && ' \
                                  f'export OUTPUT_RESULT_DIR={OUTPUT_RESULT_DIR} && '
                    cmd = config.getComplianceTest01Part3Step1Cmd(model)
                    run_bash(f'{ENV_PARAMS1} {cmd}')
                    
                    print(f'Performing compliance TEST01 part 3 step #2.')
                    cmd = config.getComplianceTest01Part3Step2Cmd(model)
                    run_bash(f'{ENV_PARAMS1} {cmd}')
                
                    print(f'Performing compliance TEST01 part 3 step #3.')
                    cmd = config.getComplianceTest01Part3Step3Cmd(model)
                    run_bash(f'{ENV_PARAMS1} {cmd}')

                if ACC_PASS & PERF_PASS:
                    result.add_compliance_test_results(model, impl, dtype, scenario, test, 'PASS')
                else:
                    result.add_compliance_test_results(model, impl, dtype, scenario, test, 'FAIL')

                # Generate accuracy.txt
                COMPLIANCE_ACC_OUTPUT_DIR = os.path.join(COMPLIANCE_OUTPUT_DIR, test, 'accuracy')
                gen_acc_cmd = config.getComplianceTest01GenAccTxtCmd(model)
                gen_acc_cmd = f'{EXPORT_ENV_PARAMS} && cd {COMPLIANCE_ACC_OUTPUT_DIR} && {gen_acc_cmd}'
                run_bash(gen_acc_cmd)
            else:
                if PERF_PASS:
                    result.add_compliance_test_results(model, impl, dtype, scenario, test, 'PASS')
                else:
                    result.add_compliance_test_results(model, impl, dtype, scenario, test, 'FAIL')

    print(f'Completed {model}/{impl}/{dtype}/{scenario}/{mode}.')

def exec_workload(model: str, impl: str, dtype: str):
    for scenario in ['Offline', 'Server']:
        if is_run_scenario(model, impl, dtype, scenario):
            # backup log
            MEASUREMENT_DIR = config.get_output_measurements_dir(model, impl, dtype, scenario)
            if os.path.exists(MEASUREMENT_DIR):
                shutil.copytree(MEASUREMENT_DIR, f'{MEASUREMENT_DIR}_{TIMESTAMP}')
            else:
                os.makedirs(MEASUREMENT_DIR)

            # perform test
            for mode in ['accuracy', 'performance', 'compliance']:
                if is_run_mode(model, impl, dtype, scenario, mode):
                    start_ts = time.time()
                    if mode == 'compliance':
                        run_compliance(model, impl, dtype, scenario, mode) 
                    else:
                        run_mode(model, impl, dtype, scenario, mode)
                    print(f"Elapsed time: {time.time() - start_ts} sec.")
                else:
                    print(f'Skipped {model}/{impl}/{dtype}/{scenario}/{mode}.')
        else:
            print(f'Skipped {model}/{impl}/{dtype}/{scenario}.')

collect_env_info()
export_env(model, impl, dtype)
try:
    prepare(model, impl, dtype)
    exec_workload(model, impl, dtype)
finally:
    result.flush()
