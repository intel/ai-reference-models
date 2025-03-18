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
at::Tensor SigmoidFocalLoss_forward(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const int num_classes, 
		const float gamma, 
		const float alpha) {
  if (logits.type().is_cuda()) {
#ifdef WITH_CUDA
    return SigmoidFocalLoss_forward_cuda(logits, targets, num_classes, gamma, alpha);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor SigmoidFocalLoss_backward(
			     const at::Tensor& logits,
                             const at::Tensor& targets,
			     const at::Tensor& d_losses,
			     const int num_classes,
			     const float gamma,
			     const float alpha) {
  if (logits.type().is_cuda()) {
#ifdef WITH_CUDA
    return SigmoidFocalLoss_backward_cuda(logits, targets, d_losses, num_classes, gamma, alpha);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
