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
# export VOCAB_FILE=/data/wiki/bert-base-uncased-vocab.txt
cpus=$( ls -d /sys/devices/system/cpu/cpu[[:digit:]]* | wc -w )
if [ ${cpus} -gt 64 ]; then
cpus=64
fi
echo "Using ${cpus} CPU cores..."
datadir=$1
find -L ${datadir} -name "pretrain-part*" | xargs --max-args=1 --max-procs=${cpus} bash create_pretraining_data.sh
