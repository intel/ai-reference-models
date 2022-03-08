@echo OFF

REM Copyright (c) 2022 Intel Corporation
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM    http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.
REM

REM uncomment the following line to enable script debugging
REM set _DEBUG_=1
REM This script sets up an optimum CPU-affinity to achieve high performance with AI benchmark workload on a few specific Intel client hardwares.
REM User would double click this batch file to open a command window with necessary settings


SETLOCAL enableextensions ENABLEDELAYEDEXPANSION

REM check if hypethreading is ON

SET count=1
FOR /F "tokens=* USEBACKQ" %%a in (`wmic cpu get numberOfLogicalProcessors`) do (
  SET tmpvar!count!=%%a
  SET /a count=!count!+1
)
SET num_logical_procs=%tmpvar2%
set num_logical_procs=%num_logical_procs: =%

SET count=1
FOR /F "tokens=* USEBACKQ" %%a in (`wmic cpu get numberOfCores`) do (
  SET tmpvar!count!=%%a
  SET /a count=!count!+1
)
SET num_physical_cores=%tmpvar2%
set num_physical_cores=%num_physical_cores: =%

REM if no hypethreading, then don't set CPU-affinity
IF %num_physical_cores%==%num_logical_procs% exit

if defined _DEBUG_ echo "num_logical_procs is %num_logical_procs%"
if defined _DEBUG_ echo "num_physical_cores is %num_physical_cores%"

REM Set appropriate affinity string based on processor info

REM check CPU id, ref https://en.wikichip.org/wiki/intel/cpuid

SET cpu_id=%PROCESSOR_IDENTIFIER%

if not "x%cpu_id:Family 6 Model 140=%"=="x%cpu_id%" set MachineTigerLake=1
if not "x%cpu_id:Family 6 Model 141=%"=="x%cpu_id%" set MachineTigerLake=1
IF defined MachineTigerLake (
if defined _DEBUG_ echo %cpu_id% = TIGERLAKE 8 logical procs
REM Hex 5555 == 0101 0101 -> TigerLake only 8 cores 1 thread/core-> CPU numbers: 0 (LSB position), 2, 4, 6
set affinity_str=55
) ELSE (
if not "x%cpu_id:Family 6 Model 151=%"=="x%cpu_id%" set MachineAlderLake=1
if not "x%cpu_id:Family 6 Model 154=%"=="x%cpu_id%" set MachineAlderLake=1
IF defined MachineAlderLake (
if defined _DEBUG_ echo %cpu_id% = ALDERLAKE
REM Hex 5555 == 0101 0101 0101 0101 -> AlderLake only 8 big cores 1 thread/core-> CPU numbers: 0 (LSB position), 2, 4, 6, 8, 10, 12, 14
set affinity_str=5555
)
)

REM Finally...
set TF_ENABLE_ONEDNN_OPTS=1
set MPI_NUM_PROCESSES=None
set MPI_HOSTNAMES=None

if defined _DEBUG_ echo affinity_str = %affinity_str%

if defined _DEBUG_ pause

IF not "x%affinity_str%"=="x" (
start /affinity %affinity_str% "TensorFlow with Intel OneDNN Optimizations & CPU-affinity" "cmd.exe"
) ELSE (
start "TensorFlow with Intel(R) oneDNN Optimizations" "cmd.exe"
)
ENDLOCAL
exit

