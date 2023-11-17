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
# ============================================================================

import json
import os
import io
import logging
import torch
import psutil

from metadata import *


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(filename)s:%(lineno)d %(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

LOG_LEVEL = int(os.environ.get("GPTJ_LOG_LEVEL", logging.INFO))
logger = logging.getLogger("GPTJLogger")
logger.setLevel(LOG_LEVEL)
logger.propagate = False
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(CustomFormatter())
    logger.addHandler(stream_handler)

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def get_memory_usage(name, device_type, device=None):
    if device_type == "cuda":
        memory_allocated = round(torch.cuda.memory_reserved(device) / 1024**3, 3)
    elif device_type == "xpu":
        memory_allocated = round(torch.xpu.memory_reserved(device) / 1024**3, 3)
    else:
        memory_allocated = round(psutil.Process().memory_info().rss / 1024**3, 3)
    logger.info(f"{name} memory used total: {memory_allocated} GB")

def profile_handler(profile_ctx, device_type="cpu", profile_prefix=None):
    logger.info(f"Exporting profile to ./profile/{profile_prefix}.prof")
    profile_table = profile_ctx.key_averages().table(
        sort_by=f"self_{device_type}_time_total", row_limit=-1)
    print(profile_table)
    with open(f"./profile/{profile_prefix}.prof", "w") as profile_file:
        profile_file.write(profile_table)

    logger.info(f"Exporting trace to ./profile/{profile_prefix}.json")
    profile_ctx.export_chrome_trace=f"./profile/{profile_prefix}.json"

def get_batch_size(input_len, is_int4=False):
    if is_int4:
        batch_size = min(INT4_SATURATE_BS, INT4_WARMUP_LEN * INT4_WARMUP_BS // input_len)
    else:
        batch_size = min(FP16_SATURATE_BS, FP16_WARMUP_LEN * FP16_WARMUP_BS// input_len)
    return batch_size
