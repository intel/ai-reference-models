#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# AGPL-3.0 license

local_dir = "./cat"
snapshot_download("diffusers/cat_toy_example", local_dir=local_dir, repo_type="dataset", ignore_patterns=".gitattributes")
