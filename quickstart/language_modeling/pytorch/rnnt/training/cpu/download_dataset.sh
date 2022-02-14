#!/usr/bin/env bash
#
# Copyright (c) 2020 Intel Corporation
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

MODEL_DIR=${MODEL_DIR-$PWD}
DATASET_DIR=${DATASET_DIR-$PWD}

cd $MODEL_DIR/models/language_modeling/pytorch/rnnt/training/cpu
dir=$(pwd)
mkdir -p ${DATASET_DIR}/dataset; cd ${DATASET_DIR}/dataset

WORKDIR=`pwd`
mkdir $WORKDIR/local
export install_dir=$WORKDIR/local
cd $WORKDIR && mkdir third_party
wget https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz -O third_party/flac-1.3.2.tar.xz
cd third_party && tar xf flac-1.3.2.tar.xz && cd flac-1.3.2
./configure --prefix=$install_dir && make && make install

cd $WORKDIR
wget https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz -O third_party/sox-14.4.2.tar.gz
cd third_party && tar zxf sox-14.4.2.tar.gz && cd sox-14.4.2
LDFLAGS="-L${install_dir}/lib" CFLAGS="-I${install_dir}/include" ./configure --prefix=$install_dir --with-flac && make &&    make install

cd $WORKDIR
wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz -O third_party/libsndfile-1.0.28.tar.gz
cd third_party && tar zxf libsndfile-1.0.28.tar.gz && cd libsndfile-1.0.28
./configure --prefix=$install_dir && make && make install

export LD_LIBRARY_PATH=$WORKDIR/local/lib:$LD_LIBRARY_PATH

cd $MODEL_DIR/models/language_modeling/pytorch/rnnt/training/cpu
python utils/download_librispeech.py utils/librispeech.csv $DATASET_DIR/dataset/ -e $DATASET_DIR/dataset/

export PATH=$WORKDIR/local/bin:$PATH
python utils/convert_librispeech.py --input_dir $DATASET_DIR/dataset/LibriSpeech/dev-clean --dest_dir $DATASET_DIR/dataset/LibriSpeech/dev-clean-wav --output_json $DATASET_DIR/dataset/LibriSpeech/librispeech-dev-clean-wav.json
python utils/convert_librispeech.py --input_dir $DATASET_DIR/dataset/LibriSpeech/train-clean-100 --dest_dir $DATASET_DIR/dataset/LibriSpeech/train-clean-100-wav  --output_json $DATASET_DIR/dataset/LibriSpeech/librispeech-train-clean-100-wav.json
python utils/convert_librispeech.py --input_dir $DATASET_DIR/dataset/LibriSpeech/train-clean-360 --dest_dir $DATASET_DIR/dataset/LibriSpeech/train-clean-360-wav  --output_json $DATASET_DIR/dataset/LibriSpeech/librispeech-train-clean-360-wav.json
python utils/convert_librispeech.py --input_dir $DATASET_DIR/dataset/LibriSpeech/train-other-500 --dest_dir $DATASET_DIR/dataset/LibriSpeech/train-other-500-wav  --output_json $DATASET_DIR/dataset/LibriSpeech/librispeech-train-other-500-wav.json

cd $dir
