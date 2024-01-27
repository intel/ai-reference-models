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
pip install -r requirements.txt

WIKI_BASE_DIR=$1

python ./data/wiki_downloader.py --language=en --save_path=${WIKI_BASE_DIR}
python ./data/WikiExtractor.py ${WIKI_BASE_DIR}wikicorpus_en/wikicorpus_en.xml -o ${WIKI_BASE_DIR}wikicorpus_en/text
cd data/wikicleaner/
bash run.sh "${WIKI_BASE_DIR}wikicorpus_en/text/*/wiki_??" ${WIKI_BASE_DIR}wikicorpus_en/results
cd ..
python vocab_downloader.py --type=bert-base-uncased
mv bert-base-uncased-vocab.txt ${WIKI_BASE_DIR}wikicorpus_en
export VOCAB_FILE=${WIKI_BASE_DIR}wikicorpus_en/bert-base-uncased-vocab.txt
bash parallel_create_pretraining_data.sh ${WIKI_BASE_DIR}wikicorpus_en/results/
