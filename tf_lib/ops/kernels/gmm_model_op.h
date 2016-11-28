//************************************************************************
//      __   __  _    _  _____   _____
//     /  | /  || |  | ||     \ /  ___|
//    /   |/   || |__| ||    _||  |  _
//   / /|   /| ||  __  || |\ \ |  |_| |
//  /_/ |_ / |_||_|  |_||_| \_\|______|
//    
// 
//   Written by: Daniel L. Marino (marinodl@vcu.edu) (2016)
//
//   Copyright (2016) Modern Heuristics Research Group (MHRG)
//   Virginia Commonwealth University (VCU), Richmond, VA
//   http://www.people.vcu.edu/~mmanic/
//   
//   This program is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   This program is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//  
//   Any opinions, findings, and conclusions or recommendations expressed 
//   in this material are those of the author's(s') and do not necessarily 
//   reflect the views of any other entity.
//  
//   ***********************************************************************
//
//   Description:   Implementation of a Gaussian Mixture Model
//
//   ***********************************************************************


#ifndef TENSORFLOW_KERNELS_GMM_MODEL_OP_H_
#define TENSORFLOW_KERNELS_GMM_MODEL_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"


#include "cuda.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
//#include "tensorflow/core/util/stream_executor_util.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h"
#include "unsupported/Eigen/CXX11/ThreadPool"
//#include "tensorflow/core/platform/stream_executor.h"


#include "twodlearn/common/cuda/matmul_pattern_cu.h"
#include <iostream>
#include <cmath>

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define PI 3.14159265358979323846

namespace tensorflow{
namespace functor {

  // definition of the functor
  template <typename Device, typename T>
    struct GmmModelFunctor {
      // Computes on device "d": p_x= sum_k (w_k * Gaussian(x_i, mu_k, sigma_k)).
      void operator()(
		      const OpKernelContext* context,
		      Tensor& p_x_tf,
		      Tensor& gaussians_tf,
		      Tensor& sigma_inv_x_mu_tf,
		      const Tensor& x_tf,
		      const Tensor& w_tf,
		      const Tensor& mu_tf,
		      const Tensor& sigma_tf
		      );
    };


} // end namespace functor
} // end namespace tensorflow

#endif // TENSORFLOW_KERNELS_GMM_MODEL_OP_H_
