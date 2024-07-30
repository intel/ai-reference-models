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
import yaml
from typing import Dict, Optional
import os.path as osp

class RuntimeConfig:
  
    _conf: Optional[Dict[str, str]]

    def __init__(self, config_path: str, logger):
        with open(config_path, 'r') as f_config:
            self._conf = yaml.safe_load(f_config)
        self.logger = logger

    def get_empty_str_if_none(self, inp) -> str:
        return str(inp or '')

    def get_version(self) -> str:
        return str(self._conf['version'])

    def get_run_interval(self) -> str:
        return str(self._conf['run_interval'])
    
    def get_org(self) -> str:
        return self._conf['org']

    def get_model_names(self) -> list:
        return list(self._conf['benchmarks'].keys())
      
    def check_model_support(self, modelName:str) -> bool:
        return True if modelName in self.get_model_names() else False
         
    def get_model(self, modelName:str):
        if self.check_model_support(modelName):
            return self._conf['benchmarks'][modelName]
        else:
            raise NotImplementedError(f'{modelName}. Please check config.yaml')

    def get_division(self, modelName) -> str:
        return self.get_model(modelName)['division']
        
    def getComplianceTests(self, modelName):
        if self.check_model_support(modelName):
            return self._conf['benchmarks'][modelName]['compliance_tests']

    def getComplianceTest01GenAccTxtCmd(self, modelName):
        if self.check_model_support(modelName):
            return self._conf['benchmarks'][modelName]['compliance_test01_gen_accuracy_txt']

    def getComplianceTest01Part3Step1Cmd(self, modelName):
        if self.check_model_support(modelName):
            return self._conf['benchmarks'][modelName]['compliance_test01_part3']['step1']['cmd']

    def getComplianceTest01Part3Step2Cmd(self, modelName):
        if self.check_model_support(modelName):
            return self._conf['benchmarks'][modelName]['compliance_test01_part3']['step2']['cmd']
        
    def getComplianceTest01Part3Step3Cmd(self, modelName):
        if self.check_model_support(modelName):
            return self._conf['benchmarks'][modelName]['compliance_test01_part3']['step3']['cmd']

    def check_model_impl_support(self, modelName:str, impl:str) -> bool:
        impls = self.get_model(modelName)['implementations']
        return True if impl in list(impls.keys()) else False

    def get_model_impl(self, modelName:str, impl:str):
        impls = self.get_model(modelName)['implementations']
        if impl in list(impls.keys()):
            return impls[impl]
        else:
            self.logger.error(f'Implementation not found in config.yaml: {modelName}/{impl}')
            raise NotImplementedError(f'{modelName}/{impl}. Please check config.yaml')

    def check_dtype_support(self, modelName:str, impl:str, dtype:str) -> bool:
        dtypes = self.get_model_impl(modelName, impl)['dtypes']
        return True if dtype in list(dtypes.keys()) else False

    def get_dtype(self, modelName:str, impl:str, dtype:str):
        dtypes = self.get_model_impl(modelName, impl)['dtypes']
        if dtype in list(dtypes.keys()):
            return dtypes[dtype]
        else:
            self.logger.error(f'dtype not found in config.yaml: {modelName}/{impl}/{dtype}')
            raise NotImplementedError(f'{modelName}/{impl}/{dtype}. Please check config.yaml')

    def get_preproc_output_dir(self, modelName:str, impl:str, dtype:str) -> str:
        return self.get_dtype(modelName, impl, dtype)['preproc_output_dir']

    def get_preproc_cmds(self, modelName:str, impl:str, dtype:str) -> str:
        return self.get_dtype(modelName, impl, dtype)['preproc_cmds']

    def get_params(self, modelName:str, impl:str, dtype:str) -> str:
        return self.get_dtype(modelName, impl, dtype)['params']
           
    def get_sys_desc(self, modelName:str, impl:str, dtype:str) -> str:
        return self.get_dtype(modelName, impl, dtype)['system_desc']
    
    def get_model_conf_path(self, modelName:str, impl:str, dtype:str) -> str:
        return self.get_empty_str_if_none(self.get_dtype(modelName, impl, dtype)['conf_path'])

    def check_scenario_support(self, modelName:str, impl:str, dtype:str, scenario:str) -> bool:
        if self.check_dtype_support(modelName, impl, dtype):
            scenarios = self.get_dtype(modelName, impl, dtype)['scenarios']
            return True if scenario in list(scenarios.keys()) else False
        else:
            return False
   
    def get_scenario(self, modelName:str, impl:str, dtype: str, scenario:str):
        scenarios = self.get_dtype(modelName, impl, dtype)['scenarios']
        if scenario in list(scenarios.keys()):
            return scenarios[scenario]
        else:
            self.logger.error(f'Scenario not found in config.yaml: {modelName}/{impl}/{dtype}/{scenario}')
            raise NotImplementedError(f'{modelName}/{impl}/{dtype}/{scenario}. Please check config.yaml')

    def check_test_mode_support(self, modelName:str, impl:str, dtype:str, scenario:str, testMode:str) -> bool:
        if self.check_scenario_support(modelName, impl, dtype, scenario):
            testModes = self.get_scenario(modelName, impl, dtype, scenario)
            return True if testMode in list(testModes.keys()) else False
        else:
            return False

    def get_test_mode(self, modelName:str, impl:str, dtype:str, scenario:str, testMode:str):
        testModes = self.get_scenario(modelName, impl, dtype, scenario)
        if testMode in list(testModes.keys()):
            return testModes[testMode]
        else:
            self.logger.error(f'Test mode not found in config.yaml: {modelName}/{impl}/{dtype}/{scenario}/{testMode}')
            raise NotImplementedError(f'{modelName}/{impl}/{dtype}/{scenario}/{testMode}. Please check config.yaml')
        
    def get_script(self, modelName:str, impl:str, dtype:str, scenario:str, testMode:str) -> str:
        return self.get_test_mode(modelName, impl, dtype, scenario, testMode)['script']
        
    def get_pre_args(self, modelName:str, impl:str, dtype:str, scenario:str, testMode:str) -> str:
        return self.get_test_mode(modelName, impl, dtype, scenario, testMode)['pre_args']

    def get_post_args(self, modelName:str, impl:str, dtype:str, scenario:str, testMode:str) -> str:
        return self.get_test_mode(modelName, impl, dtype, scenario, testMode)['post_args']

    def get_postproc_cmds(self, modelName:str, impl:str, dtype:str, scenario:str, testMode:str) -> str:
        return self.get_test_mode(modelName, impl, dtype, scenario, testMode)['postproc_cmds']

    def get_source(self, modelName:str, impl:str, dtype:str, scenario:str, testMode:str) -> str:
        return self.get_test_mode(modelName, impl, dtype, scenario, testMode)['source']
        
    def get_exec_cmd(self, modelName:str, impl:str, dtype:str, scenario:str, testMode:str) -> str:
        source = self.get_source(modelName, impl, dtype, scenario, testMode)
        source = f"source {source} &&" if source else ""
        preArgs = self.get_empty_str_if_none(self.get_pre_args(modelName, impl, dtype, scenario, testMode))
        postArgs = self.get_empty_str_if_none(self.get_post_args(modelName, impl, dtype, scenario, testMode))
        script = self.get_script(modelName, impl, dtype, scenario, testMode)
        postproc_cmds = self.get_postproc_cmds(modelName, impl, dtype, scenario, testMode)
        postproc_cmds = f"&& {postproc_cmds}" if postproc_cmds else ""
        return f" pwd && {source} {preArgs} bash {script} {postArgs} {postproc_cmds}"
    
    def get_container_impl(self, modelName:str, impl:str, dtype:str):
        implName = f'{modelName}/{impl}/{dtype}'
        if implName in list(self._conf['container'].keys()):
            return self._conf['container'][implName]
        else:
            self.logger.error(f'container configuration not found in config.yaml: {implName}')
            raise NotImplementedError(f'{implName}. Please check config.yaml')   

    def get_docker_image_name(self, modelName:str, impl:str, dtype:str) -> str:
        return self.get_container_impl(modelName, impl, dtype)['image_name']         

    def get_container_output_dir(self, modelName:str, impl:str, dtype:str) -> str:
        return self.get_container_impl(modelName, impl, dtype)['output_dir']

    def get_container_compliance_suite_dir(self, modelName:str, impl:str, dtype:str) -> str:
        return self.get_container_impl(modelName, impl, dtype)['compliance_suite_dir']

    def get_compliance_test01_model_name(self, modelName:str, impl:str, dtype:str) -> str:
        return self.get_container_impl(modelName, impl, dtype)['compliance_test01_model_name']

    def get_container_workdir(self, modelName:str, impl:str, dtype:str) -> str:
        return self.get_container_impl(modelName, impl, dtype)['work_dir']
    
    def get_container_automation_dir(self, modelName:str, impl:str, dtype:str) -> str:
        return f"{self.get_container_impl(modelName, impl, dtype)['work_dir']}/automation"

    def get_container_data_dir(self, modelName:str, impl:str, dtype:str) -> str:
        return self.get_container_impl(modelName, impl, dtype)['data_dir']   

    def get_container_model_dir(self, modelName:str, impl:str, dtype:str) -> str:
        return self.get_container_impl(modelName, impl, dtype)['model_dir']

    # def get_container_code_dir(self, modelName:str, impl:str, dtype:str) -> str:
    #     return osp.join(self.get_container_workdir(modelName, impl, dtype),
    #                     'code', modelName, impl)

    def get_image_builder(self, modelName:str, impl:str, dtype:str) -> str:  
        return self.get_container_impl(modelName, impl, dtype)['image_builder']

    def get_container_log_dir(self, modelName:str, impl:str, dtype:str, scenario:str, testMode:str) -> str:
        return self.get_empty_str_if_none(
            self.get_test_mode(modelName, impl, dtype, scenario, testMode)['log_dir']
        )
        
    def get_output_result_dir(self, modelName:str, impl:str, dtype:str):
        return osp.join(self.get_container_output_dir(modelName, impl, dtype),
                        self.get_division(modelName),
                        self.get_org(),
                        'results',
                        self.get_sys_desc(modelName, impl, dtype),
                        modelName)
    
    def get_output_compliance_dir(self, modelName:str, impl:str, dtype:str, scenario:str):
        return osp.join(self.get_container_output_dir(modelName, impl, dtype),
                        self.get_division(modelName),
                        self.get_org(),
                        'compliance',
                        self.get_sys_desc(modelName, impl, dtype),
                        modelName,
                        scenario)
    
    def get_output_measurements_dir(self, modelName:str, impl:str, dtype:str, scenario:str):
        return osp.join(self.get_container_output_dir(modelName, impl, dtype),
                        self.get_division(modelName),
                        self.get_org(),
                        'measurements',
                        self.get_sys_desc(modelName, impl, dtype),
                        modelName,
                        scenario)
