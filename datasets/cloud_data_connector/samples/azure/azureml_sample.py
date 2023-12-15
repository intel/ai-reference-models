#
# -*- coding: utf-8 -*-
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
# This sample is based on azure ML official tutorial
#  https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-azure-ml-in-a-day
#
# Python
from pathlib import Path
# AZURE
# Azure ML
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    AmlCompute,
    Environment,
    Workspace
)

# Data connector
from cloud_data_connector.azure import Connector, MLUploader

#  Connect to AzureML
# This connector requires config.json file with your account values set
ml_client: MLClient = Connector().connect()
ml_uploader: MLUploader = MLUploader(ml_client)
# Data connector helps to train models in local
# After training is possible upload training code,
# model and set configurations for a workspace and 
# clusters

# Training the model in local
path = Path().resolve()
training_script = f"{path}/src/main.py"
training_data = f"{path}/sample_data/credit_card_clients.xls"
_test_train_ratio=0.2
_learning_rate=0.25
_registered_model_name = "credit_defaults_model"
# Here is our training code ./src/main.py
from src.main import main
arguments = [
        "--data", f"{training_data}",
        "--test_train_ratio", f"{_test_train_ratio}",
        "--learning_rate", f"{_learning_rate}",
        "--registered_model_name", f"{_registered_model_name}"
        ]
main(arguments)

# After training new folders are created:
# ./credit_defaults_model
# ./mlruns
# Those folders will help us to follow an standar workflow in our 
# ML process. 

# AZURE ML Cloud
# 
# AzureML requires some simple configurations, is 
# simple to upload configurations usig AzuireML library or 
# Data connector. Data connector helps to handle errors and you can re use 
# same connector strategy to move into blobs

# For tis sample we will create a Compute resource (AmlCompute),
# An environment and the job.


# AmlCompute 
# Name assigned to the compute cluster
cpu_compute_target = "cpu-cluster"
_type = "amlconpute"
_size = "STANDARD_DS3_V2"
_min_instances = 0
_max_intances = 2
_idle_time_before_scale_down = 180
_tier = "Dedicated"
try:
    # let's see if the compute target already exists
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
    print(
        f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    # Let's create the Azure ML compute object with the intended parameters
    cpu_cluster = AmlCompute(
        name=cpu_compute_target,
        # Azure ML Compute is the on-demand VM service
        type=_type,
        # VM Family
        size=_size,
        # Minimum running nodes when there is no job running
        min_instances=_min_instances,
        # Nodes in cluster
        max_instances=_max_intances,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=_idle_time_before_scale_down,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier=_tier,
    )
    print(
        f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}"
    )
    # Now, we pass the object to MLClient's create_or_update method
    cpu_cluster = ml_uploader.upload(cpu_cluster)

#Environment
# 
dependencies_dir = f"{path}/dependencies/" 
custom_env_name =  "aml-scikit-learn"
_environment_description = (
                            "Custom environment for Credit Card " 
                            "Defaults pipeline"
                        )
_environment_tags = {"scikit-learn": "0.24.2"}
_environment_conda_file = f"{dependencies_dir}/conda.yml"
_environment_docker_image = "mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest"

pipeline_job_env: Environment = Environment(
    name=custom_env_name,
    description=_environment_description,
    tags=_environment_tags,
    conda_file=_environment_conda_file,
    image=_environment_docker_image,
)
# Upload environment using data connector
pipeline_job_env = ml_uploader.upload(pipeline_job_env)

print(
    f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
)

# Job
# Now we will set a job to run on cloud, for this moment we know 
# the script to training and we have a data set for training
# 

from azure.ai.ml import command
from azure.ai.ml import Input

# We shoudl define some extra values to run in cloud systems
_code = ".src"
_comand = """python main.py --data ${{inputs.data}} 
        --test_train_ratio ${{inputs.test_train_ratio}} 
        --learning_rate ${{inputs.learning_rate}} 
        --registered_model_name ${{inputs.registered_model_name}}
        """
_job_environment="aml-scikit-learn@latest"
_job_compute="cpu-cluster"
_job_experiment_name="train_model_credit_default_prediction"
_job_display_name="credit_default_prediction_from_data_connector"

job = command(
    inputs=dict(
        data=Input(
            type="uri_file",
            path=training_data,
        ),
        test_train_ratio=_test_train_ratio,
        learning_rate=_learning_rate,
        registered_model_name=_registered_model_name,
    ),
    code="./src/",  # location of source code
    command="python main.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --learning_rate ${{inputs.learning_rate}} --registered_model_name ${{inputs.registered_model_name}}",
    environment=_job_environment,
    compute=_job_compute,
    experiment_name=_job_experiment_name,
    display_name=_job_display_name,
)

ml_uploader.upload(job)

# Now you can see the training running on wour AzureML workspace dashboard