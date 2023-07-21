
import argparse
import yaml
import subprocess
from slugify import slugify
import os
from collections import namedtuple

TestResult = namedtuple('TestResult', 'test_name status log_file docker_cmd')

def run_command(command, log_file_path):
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(command.split(), stdout=log_file, stderr=log_file)
        process.wait()
    status = "SUCCESS" if process.returncode == 0 else "FAILED"
    return status

def main(tests_yaml, logs_dir):
    print(tests_yaml, logs_dir)
    with open(tests_yaml, "r") as stream:
        try:
            workloads = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    results = {}
    for workload_name, workload_value in workloads.items():
        print("workload_name: ", workload_name)
        print(" image_name: ", workload_value['image_name'])
        image_name = workload_value['image_name']
        test_results = []
        for test in workload_value['tests']:
            test_name = slugify(test['test_name'])
            log_file = os.path.join(logs_dir,
                                    slugify(test['test_name'])+".log")
            env_vars = test['env_vars'] #its list not dict
            volumes = test['volumes']

            docker_cmd = "docker run --rm " + env_vars['DOCKER_ARGS']
            docker_cmd += " --env http_proxy={}".format(os.environ['http_proxy'])
            docker_cmd += " --env https_proxy={}".format(os.environ['http_proxy'])
            docker_cmd += " --env no_proxy={}".format(os.environ['no_proxy']) 
            for key, value in env_vars.items():
                docker_cmd += " --env {}={}".format(key,value)
            #TODO: Bug - what if volume directory is does not exists, docker test will
            #      eventually fail but test status will be success as docker cmd
            #      will return success
            for key, value in volumes.items():
                docker_cmd += " --env {}={}".format(key, value)
            for key, value in volumes.items():
                docker_cmd += " --volume {}:{}".format(value, value)
            docker_cmd += " " + image_name
            docker_cmd += " /bin/bash " + env_vars['SCRIPT']

            #print("docker cmd: ", docker_cmd)
            status = run_command(docker_cmd, log_file)
            test_results.append(TestResult(test_name, status, log_file, docker_cmd))
            print("# ",slugify(test['test_name']), " - ", status, " - ", log_file)
        results[workload_name] = test_results
    # TODO: summerize all results
    # with open(os.path.join(logs_dir, 'results.txt'), 'w') as f:
    #     f.write("---- Test results ----\n")
    #     for workload, test_results in results.items():
    #         f.write(F"\n{workload} \n")
    #         for test in test_results:
    #             f.write(F"  {test.test_name} - {test.status}  - { test.log_file}  - {test.docker_cmd}\n")

if __name__ == "__main__":
    print("---- Test Runner --")
    parser = argparse.ArgumentParser(description="Test runner execute tests run command and capture logs")

    parser.add_argument("--tests-yaml", type=str, dest="tests_yaml", required=True,
                        help="Yaml file specifying workload image and its tests")
    parser.add_argument("--logs-dir", dest="logs_dir",
                        default="./logs", type=str,
                        help="Path to logs directory")

    args = parser.parse_args()
    os.makedirs(os.path.abspath(args.logs_dir),
                           exist_ok=True)

    main(os.path.abspath(args.tests_yaml), 
         os.path.abspath(args.logs_dir))
