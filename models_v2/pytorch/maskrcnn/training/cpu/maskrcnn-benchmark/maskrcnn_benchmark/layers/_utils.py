# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import glob
# MIT License
# 
# Copyright (c) 2018 Facebook
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os.path

import torch

try:
    from torch.utils.cpp_extension import load as load_ext
    from torch.utils.cpp_extension import CUDA_HOME
except ImportError:
    raise ImportError("The cpp layer extensions requires PyTorch 0.4 or higher")


def _load_C_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir = os.path.dirname(this_dir)
    this_dir = os.path.join(this_dir, "csrc")

    main_file = glob.glob(os.path.join(this_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(this_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(this_dir, "cuda", "*.cu"))

    source = main_file + source_cpu

    extra_cflags = []
    if torch.cuda.is_available() and CUDA_HOME is not None:
        source.extend(source_cuda)
        extra_cflags = ["-DWITH_CUDA"]
    source = [os.path.join(this_dir, s) for s in source]
    extra_include_paths = [this_dir]
    return load_ext(
        "torchvision",
        source,
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
    )


_C = _load_C_extensions()
