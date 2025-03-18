// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
/*
MIT License

Copyright (c) 2018 Facebook

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


// Interface for Python
void deform_psroi_pooling_forward(
    at::Tensor input, 
    at::Tensor bbox, 
    at::Tensor trans, 
    at::Tensor out,
    at::Tensor top_count, 
    const int no_trans, 
    const float spatial_scale,
    const int output_dim, 
    const int group_size, 
    const int pooled_size,
    const int part_size, 
    const int sample_per_part, 
    const float trans_std)
{
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return deform_psroi_pooling_cuda_forward(
        input, bbox, trans, out, top_count, 
        no_trans, spatial_scale, output_dim, group_size,
        pooled_size, part_size, sample_per_part, trans_std
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}


void deform_psroi_pooling_backward(
    at::Tensor out_grad, 
    at::Tensor input, 
    at::Tensor bbox, 
    at::Tensor trans,
    at::Tensor top_count, 
    at::Tensor input_grad, 
    at::Tensor trans_grad,
    const int no_trans, 
    const float spatial_scale, 
    const int output_dim,
    const int group_size, 
    const int pooled_size, 
    const int part_size,
    const int sample_per_part, 
    const float trans_std) 
{
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return deform_psroi_pooling_cuda_backward(
        out_grad, input, bbox, trans, top_count, input_grad, trans_grad,
        no_trans, spatial_scale, output_dim, group_size, pooled_size, 
        part_size, sample_per_part, trans_std
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
