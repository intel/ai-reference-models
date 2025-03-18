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
int deform_conv_forward(
    at::Tensor input, 
    at::Tensor weight,
    at::Tensor offset, 
    at::Tensor output,
    at::Tensor columns, 
    at::Tensor ones, 
    int kW,
    int kH, 
    int dW, 
    int dH, 
    int padW, 
    int padH,
    int dilationW, 
    int dilationH, 
    int group,
    int deformable_group, 
    int im2col_step)
{
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return deform_conv_forward_cuda(
        input, weight, offset, output, columns, ones,
        kW, kH, dW, dH, padW, padH, dilationW, dilationH,
        group, deformable_group, im2col_step
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}


int deform_conv_backward_input(
    at::Tensor input, 
    at::Tensor offset,
    at::Tensor gradOutput, 
    at::Tensor gradInput,
    at::Tensor gradOffset, 
    at::Tensor weight,
    at::Tensor columns, 
    int kW, 
    int kH, 
    int dW,
    int dH, 
    int padW, 
    int padH, 
    int dilationW,
    int dilationH, 
    int group,
    int deformable_group, 
    int im2col_step)
{
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return deform_conv_backward_input_cuda(
        input, offset, gradOutput, gradInput, gradOffset, weight, columns,
        kW, kH, dW, dH, padW, padH, dilationW, dilationH, 
        group, deformable_group, im2col_step
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}


int deform_conv_backward_parameters(
    at::Tensor input, 
    at::Tensor offset, 
    at::Tensor gradOutput,
    at::Tensor gradWeight,  // at::Tensor gradBias,
    at::Tensor columns, 
    at::Tensor ones, 
    int kW, 
    int kH, 
    int dW, 
    int dH,
    int padW, 
    int padH, 
    int dilationW, 
    int dilationH, 
    int group,
    int deformable_group, 
    float scale, 
    int im2col_step)
{
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return deform_conv_backward_parameters_cuda(
        input, offset, gradOutput, gradWeight, columns, ones,
        kW, kH, dW, dH, padW, padH, dilationW, dilationH,
        group, deformable_group, scale, im2col_step
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}


void modulated_deform_conv_forward(
    at::Tensor input, 
    at::Tensor weight, 
    at::Tensor bias, 
    at::Tensor ones,
    at::Tensor offset, 
    at::Tensor mask, 
    at::Tensor output, 
    at::Tensor columns,
    int kernel_h, 
    int kernel_w, 
    const int stride_h, 
    const int stride_w,
    const int pad_h, 
    const int pad_w, 
    const int dilation_h,
    const int dilation_w, 
    const int group, 
    const int deformable_group,
    const bool with_bias)
{
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return modulated_deform_conv_cuda_forward(
        input, weight, bias, ones, offset, mask, output, columns,
        kernel_h, kernel_w, stride_h, stride_w, 
        pad_h, pad_w, dilation_h, dilation_w,
        group, deformable_group, with_bias
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}


void modulated_deform_conv_backward(
    at::Tensor input, 
    at::Tensor weight, 
    at::Tensor bias, 
    at::Tensor ones,
    at::Tensor offset, 
    at::Tensor mask, 
    at::Tensor columns,
    at::Tensor grad_input, 
    at::Tensor grad_weight, 
    at::Tensor grad_bias,
    at::Tensor grad_offset, 
    at::Tensor grad_mask, 
    at::Tensor grad_output,
    int kernel_h, 
    int kernel_w, 
    int stride_h, 
    int stride_w, 
    int pad_h,
    int pad_w, 
    int dilation_h, 
    int dilation_w, 
    int group, 
    int deformable_group,
    const bool with_bias)
{
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return modulated_deform_conv_cuda_backward(
        input, weight, bias, ones, offset, mask, columns, 
        grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        group, deformable_group, with_bias
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
