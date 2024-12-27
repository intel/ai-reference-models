#!/bin/bash
# Copyright (c) 2024 Intel Corporation
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
# ============================================================================

#!/bin/bash
echo $PWD

if [ -d "jemalloc" ]; then
    rm -rf "jemalloc"
    unset LD_PRELOAD
    unset MALLOC_CONF
fi

git clone https://github.com/jemalloc/jemalloc.git
cd jemalloc || exit
git checkout c8209150f9d219a137412b06431c9d52839c7272
./autogen.sh
./configure --prefix="$PWD/"
make
make install
cd ..


if [ -d "gperftools-2.7.90" ]; then
    rm -rf "gperftools-2.7.90"
fi

wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz
tar -xzf gperftools-2.7.90.tar.gz
cd "gperftools-2.7.90" || exit
cd ..
./configure --prefix="$PWD/tcmalloc"
make
make install
cd ..
