#!/usr/bin/env bash
#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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


if [[ -z "$TMPDIR" ]]; then
	BATS_TMPDIR='/tmp'
else
	BATS_TMPDIR="${TMPDIR%/}"
fi

BATS_TMPNAME="$BATS_RUN_TMPDIR/bats.$$"
BATS_PARENT_TMPNAME="$BATS_RUN_TMPDIR/bats.$PPID"
# shellcheck disable=SC2034
BATS_OUT="${BATS_TMPNAME}.out" # used in bats-exec-file

bats_preprocess_source() {
	BATS_TEST_SOURCE="${BATS_TMPNAME}.src"
	bats-preprocess "$BATS_TEST_FILENAME" >"$BATS_TEST_SOURCE"
	trap 'bats_cleanup_preprocessed_source' ERR EXIT
	trap 'bats_cleanup_preprocessed_source; exit 1' INT
}

bats_cleanup_preprocessed_source() {
	rm -f "$BATS_TEST_SOURCE"
}

bats_evaluate_preprocessed_source() {
	if [[ -z "$BATS_TEST_SOURCE" ]]; then
		BATS_TEST_SOURCE="${BATS_PARENT_TMPNAME}.src"
	fi
	# Dynamically loaded user files provided outside of Bats.
	# shellcheck disable=SC1090
	source "$BATS_TEST_SOURCE"
}
