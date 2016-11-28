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
//   Description:   Basic matrix multiplication
//
//   ***********************************************************************


#ifndef TENSORFLOW_KERNELS_MY_MATMUL_OP_H_
#define TENSORFLOW_KERNELS_MY_MATMUL_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "twodlearn/common/cuda/matmul_pattern_cu.h"
#include <iostream>
//#include "tensorflow/core/framework/tensor_types.h"
//#include "tensorflow/core/framework/register_types.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow{
namespace functor {

  // definition of the functor
  template <typename Device, typename T>
    struct MyMatmulFunctor {
      // Computes on device "d": out = in0 * in1, where * is matrix
      // multiplication.
      void operator()(
		      //const Device& d,
		      const OpKernelContext* context,
		      Tensor& out,
		      const Tensor& in0,
		      const Tensor& in1
		      );
    };


} // end namespace functor
} // end namespace tensorflow

#endif // TENSORFLOW_KERNELS_MY_MATMUL_OP_H_
