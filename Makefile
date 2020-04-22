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

all: venv

# we need to update pip and setuptools because venv versions aren't latest
# need to prepend $(ACTIVATE) everywhere because all make calls are in subshells
# otherwise we won't be installing anything in the venv itself
$(ACTIVATE): tests/requirements.txt
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE) && python -m pip install -r requirements-test.txt
	@touch $(ACTIVATE)

venv: $(ACTIVATE)
	@echo -n "Using "
	@. $(ACTIVATE) && python --version

venv2: venv

venv3: PY_VERSION=3
venv3: $(ACTIVATE)
	@echo -n "Using "
	@. $(ACTIVATE) && python3 --version

tox:
	tox

lint:
	@echo "Running style check..."
	tox -e py2.7-flake8 -e py3-flake8

lint2:
	@echo "Running style check..."
	tox -e py2.7-flake8

lint3:
	@echo "Running style check python 3 "
	tox -e py3-flake8

unit_test:
	@echo "Running unit tests..."
	tox -e py2.7-py.test -e py3-py.test

unit_test2:
	@echo "Running unit tests python 2..."
	tox -e py2.7-py.test

unit_test3:
	@echo "Running unit tests python 3..."
	tox -e py3-py.test

test: lint unit_test

test2: lint2 unit_test2

test3: lint3 unit_test3

clean:
	rm -rf .venv .venv3 .tox
