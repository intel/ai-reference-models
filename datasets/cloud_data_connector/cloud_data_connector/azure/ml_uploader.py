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
# Azure Entities
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Job,
    Model,
    Environment,
    Component,
    Datastore,
    Workspace,
    AmlCompute,
    KubernetesCompute,
    VirtualMachineCompute,
    Compute
)

# Data typing
from typing import Union, Optional
"""
entity = Union[Job, Model, Environment, Component, Datastore]
entity = Union[entity, Workspace, AmlCompute, KubernetesCompute] 
entity = Union[entity, VirtualMachineCompute, Compute] 
"""
NOT_SUPPORTED_ENTITY = f"Object to upload ins not in list o supported entities"

class MLUploader():
    def __init__(self, connector: MLClient) -> None:
        self.connector = connector

    def upload(self, entity_obj: object)->object:
        """
        Upload entities as Job and compute resources configuration
        :param entity_obj: The entoty to load into AzureML
        """
        try:
            _ = None
            if isinstance(
                entity_obj, (Job, Model, Environment, Component, Datastore)
            ):
                _ = self.connector.create_or_update(entity_obj)
                return _
            elif isinstance(
                entity_obj,
                (AmlCompute, KubernetesCompute, VirtualMachineCompute, Compute)
            ):
                _ = self.connector.compute.begin_create_or_update(entity_obj)
                return _
            else:
                raise ValueError(f"{NOT_SUPPORTED_ENTITY} entity type: {type(entity_obj)}")

    
        except RuntimeError as rte:
            print(rte)
        
