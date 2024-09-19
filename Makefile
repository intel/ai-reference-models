# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


PY_VERSION := 2
# if py_version is 2, use virtualenv, else python3 venv
VIRTUALENV_EXE=$(if $(subst 2,,$(PY_VERSION)),python3 -m venv,virtualenv)
VIRTUALENV_DIR=$(if $(subst 2,,$(PY_VERSION)),.venv3,.venv)
ACTIVATE="$(VIRTUALENV_DIR)/bin/activate"

.PHONY: venv

all: venv lint unit_test

# we need to update pip and setuptools because venv versions aren't latest
# need to prepend $(ACTIVATE) everywhere because all make calls are in subshells
# otherwise we won't be installing anything in the venv itself
$(ACTIVATE):
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE) && python -m pip install -r requirements.txt
	@touch $(ACTIVATE)

venv: $(ACTIVATE)
	@echo -n "Using "
	@. $(ACTIVATE) && python --version

venv3: PY_VERSION=3
venv3: $(ACTIVATE)
	@echo -n "Using "
	@. $(ACTIVATE) && python3 --version

tox:
	tox

lint:
	@echo "Running style check..."
	tox -e py3-flake8

unit_test:
	@echo "Running unit tests..."
	tox -e py3-py.test

test_tl_tf_notebook: venv
	@. $(ACTIVATE) && poetry install --file docs/notebooks/transfer_learning/pyproject.toml && \
	bash docs/notebooks/transfer_learning/run_tl_notebooks.sh $(CURDIR)/docs/notebooks/transfer_learning/text_classification/tfhub_bert_text_classification/BERT_Binary_Text_Classification.ipynb remove_for_custom_dataset && \
	bash docs/notebooks/transfer_learning/run_tl_notebooks.sh $(CURDIR)/docs/notebooks/transfer_learning/image_classification/tf_image_classification/Image_Classification_Transfer_Learning.ipynb remove_for_custom_dataset && \
	bash docs/notebooks/transfer_learning/run_tl_notebooks.sh $(CURDIR)/docs/notebooks/transfer_learning/image_classification/huggingface_image_classification/HuggingFace_Image_Classification_Transfer_Learning.ipynb && \
	bash docs/notebooks/transfer_learning/run_tl_notebooks.sh $(CURDIR)/docs/notebooks/transfer_learning/question_answering/BERT_Question_Answering.ipynb && \
	bash docs/notebooks/transfer_learning/run_tl_notebooks.sh $(CURDIR)/docs/notebooks/transfer_learning/text_classification/tfhub_bert_text_classification/BERT_Multi_Text_Classification.ipynb remove_for_custom_dataset

test_tl_pyt_notebook: venv
	@. $(ACTIVATE) && poetry install --file docs/notebooks/transfer_learning/pyproject.toml && \
	bash docs/notebooks/transfer_learning/run_tl_notebooks.sh $(CURDIR)/docs/notebooks/transfer_learning/image_classification/pytorch_image_classification/PyTorch_Image_Classification_Transfer_Learning.ipynb remove_for_custom_dataset && \
	bash docs/notebooks/transfer_learning/run_tl_notebooks.sh $(CURDIR)/docs/notebooks/transfer_learning/object_detection/pytorch_object_detection/PyTorch_Object_Detection_Transfer_Learning.ipynb remove_for_custom_dataset && \
	bash docs/notebooks/transfer_learning/run_tl_notebooks.sh $(CURDIR)/docs/notebooks/transfer_learning/text_classification/pytorch_text_classification/PyTorch_Text_Classifier_fine_tuning.ipynb remove_for_custom_dataset

test: lint unit_test

clean:
	rm -rf .venv .venv3 .tox
