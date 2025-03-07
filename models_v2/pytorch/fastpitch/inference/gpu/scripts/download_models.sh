#!/usr/bin/env bash

# BSD 3-Clause License

# Copyright (c) 2020, NVIDIA Corporation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set -e

MODEL_NAMES="$@"
[ -z "$MODEL_NAMES" ] && { echo "Usage: $0 [fastpitch|waveglow|hifigan|hifigan-finetuned-fastpitch]"; exit 1; }

function download_ngc_model() {
  mkdir -p "$MODEL_DIR"

  if [ ! -f "${MODEL_DIR}/${MODEL_ZIP}" ]; then
    echo "Downloading ${MODEL_ZIP} ..."
    wget --content-disposition -O ${MODEL_DIR}/${MODEL_ZIP} ${MODEL_URL} \
         || { echo "ERROR: Failed to download ${MODEL_ZIP} from NGC"; exit 1; }
  fi

  if [ ! -f "${MODEL_DIR}/${MODEL}" ]; then
    echo "Extracting ${MODEL} ..."
    unzip -qo ${MODEL_DIR}/${MODEL_ZIP} -d ${MODEL_DIR} \
          || { echo "ERROR: Failed to extract ${MODEL_ZIP}"; exit 1; }

    echo "OK"

  else
    echo "${MODEL} already downloaded."
  fi

}

for MODEL_NAME in $MODEL_NAMES
do
  case $MODEL_NAME in
    "fastpitch")
      MODEL_DIR="pretrained_models/fastpitch"
      MODEL_ZIP="fastpitch_pyt_fp32_ckpt_v1_1_21.05.0.zip"
      MODEL="nvidia_fastpitch_210824.pt"
      MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/fastpitch_pyt_fp32_ckpt_v1_1/versions/21.05.0/zip"
      ;;
    "hifigan")
      MODEL_DIR="pretrained_models/hifigan"
      MODEL_ZIP="hifigan__pyt_ckpt_ds-ljs22khz_21.08.0_amp.zip"
      MODEL="hifigan_gen_checkpoint_6500.pt"
      MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/dle/hifigan__pyt_ckpt_ds-ljs22khz/versions/21.08.0_amp/zip"
      ;;
    "hifigan-finetuned-fastpitch")
      MODEL_DIR="pretrained_models/hifigan"
      MODEL_ZIP="hifigan__pyt_ckpt_mode-finetune_ds-ljs22khz_21.08.0_amp.zip"
      MODEL="hifigan_gen_checkpoint_10000_ft.pt"
      MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/dle/hifigan__pyt_ckpt_mode-finetune_ds-ljs22khz/versions/21.08.0_amp/zip"
      ;;
    "waveglow")
      MODEL_DIR="pretrained_models/waveglow"
      MODEL_ZIP="waveglow_ckpt_amp_256_20.01.0.zip"
      MODEL="nvidia_waveglow256pyt_fp16.pt"
      MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_amp_256/versions/20.01.0/zip"
      ;;
    *)
      echo "Unrecognized model: ${MODEL_NAME}"
      exit 2
      ;;
  esac
  download_ngc_model "$MODEL_DIR" "$MODEL_ZIP" "$MODEL" "$MODEL_URL"
done
