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
//   Description:     basic matrix multiplication operation,
//   
//   code derived from: 
//      - https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/core/kernels/matmul_op.cc
//      - https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/core/kernels/matmul_op.h
//   
//   TensorFlow uses the tensor functions provided by eigen,
//   the documentation is found in: 
//          https://bitbucket.org/eigen/eigen/src/default/unsupported/Eigen/CXX11/src/Tensor/README.md?fileviewer=file-view-default#markdown-header-controlling-how-expressions-are-evaluated
//
//
//   ***********************************************************************

#define EIGEN_USE_THREADS

#include "my_matmul_op.h"

#define BLOCK_SIZE 32

namespace tensorflow{

namespace functor {
  
  // Definition of the functor for each device
  
  // when the functor is defined, all T are replaced by the given "class"
  // dev specifies the device where the operation will be executed
  template <typename T> 
  struct MyMatmulFunctor<CPUDevice, T> {
    void operator()( const OpKernelContext* context,
		     Tensor& out,
		     const Tensor& in0,
		     const Tensor& in1){
      
      const CPUDevice d = context->eigen_device<CPUDevice>(); // get device

      // Get inputs as eigen tensors
      auto a_mat = in0.tensor<T, 2>();
      auto b_mat = in1.tensor<T, 2>();  
      // Get outputs as eigen tensors
      auto c_mat = out.tensor<T, 2>();
    
      // compute matrix multiplication using a tensor contraction, 
      // which is a generalization of the matrix product to the
      // multidimentional case 
      //Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair; // = { Eigen::IndexPair(0, 1) };
      //dim_pair[0].first = 1;
      //dim_pair[0].second = 0;
      
      Eigen::array<Eigen::IndexPair<int>, 1> dim_pair = { Eigen::IndexPair<int>(1, 0) };
    
      // out.device(dev) explicitly tells eigen the device where to execute the operation,
      // ussually, eigen functions for tensors are implemented to be run in DefaultDevice, 
      // ThreadPoolDevice and GpuDevice
      //      c_mat.device(dev) = a_mat.contract(b_mat, dim_pair);
      //c_mat = a_mat;
      c_mat.device(d) = a_mat.contract(b_mat, dim_pair);
    
    }      
  };
  
  // GPU implementation
  template <typename T>
  struct MyMatmulFunctor<GPUDevice, T>{
    void operator()( const OpKernelContext* context,
		     Tensor& out,
		     const Tensor& in0,
		     const Tensor& in1){
      
      // get data pointers
      auto in0_ptr = in0.template flat<T>().data();
      auto in1_ptr = in1.template flat<T>().data();
      auto out_ptr = out.template flat<T>().data();
      
      // define device functors
      MulFunc<T> mul_cu;
      SumFunc<T> sum_cu;
      
      // define grid dimentions and block dimentions
      dim3 dim_grid( 1 + ((out.dim_size(1) -1)/BLOCK_SIZE),
		     1 + ((out.dim_size(0) -1)/BLOCK_SIZE),
		     1);
      dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE,1);
      //std::cout << dim_grid.x << " " << dim_grid.y << " " << BLOCK_SIZE << std::endl;
      
      // run device kernel
      matmul_pattern_cuda<MulFunc<T>, SumFunc<T>, T, BLOCK_SIZE> <<<dim_grid, dim_block>>>(out_ptr, in0_ptr, in1_ptr, in0.dim_size(0), in0.dim_size(1), in1.dim_size(1), mul_cu, sum_cu);
      

      /*
      // Get inputs/outputs as eigen tensors
      auto a_mat = in0.tensor<T, 2>();
      auto b_mat = in1.tensor<T, 2>();  
      auto c_mat = out.tensor<T, 2>();  

      std::cout << "A size:" << a_mat.size() << ", dim: " << int(in0.dim_size(0)) << ", " << in0.dim_size(1) << std::endl;
      std::cout << "B size:" << b_mat.size() << ", dim: " << int(in1.dim_size(0)) << ", " << in1.dim_size(1) << std::endl;
      std::cout << "C size:" << c_mat.size() << ", dim: " << int(out.dim_size(0)) << ", " << out.dim_size(1) << std::endl;
      */
    }
  };
    
} // end namespace functor
  



// Device: DEVICE_CPU, DEVICE_GPU
// T: type of the matrix elements: float, or double
// USE_CUBLAS: true for gpu, false for cpu
template <typename Device, typename T, bool USE_CUBLAS>
class MyMatmulOp : public OpKernel {
public:
  explicit MyMatmulOp(OpKernelConstruction* context) : OpKernel(context) {}
  
  void Compute(OpKernelContext* context) override {
    // get input tensors
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    //auto input = input_tensor.flat<int32>();

    // check tensors are matrices
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(b.shape()),
		errors::InvalidArgument("In[1] is not a matirx"));
    
    // check dimentions are valid for matrix multiplication
    OP_REQUIRES(context,
                a.dim_size(1) == b.dim_size(0),
                errors::InvalidArgument("Matrix size-compatible: In[0]: ",
                                        a.shape().DebugString(), ", In[1]: ",
					b.shape().DebugString()));

    // obtain the dimentions for the output matrix
    TensorShape out_shape({a.dim_size(0), b.dim_size(1)});
    
    // Allocate the output tensor, this tells tensorflow
    // that output corresponds to the output id 0
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output));
    
    
    
    // call the function 
    functor::MyMatmulFunctor<Device, T> functor_launch;
    //functor_launch(context->eigen_device<Device>(), *output, a, b);
    functor_launch(context, *output, a, b);
    
  }
};


// Register the operations
REGISTER_OP("MyMatmul")
.Input("a: T")
.Input("b: T")
.Output("output: T")
.Attr("T: {float, double}")
.Doc(R"doc(my implementation of matrix multiplication c= a*b)doc");

// Register for float
REGISTER_KERNEL_BUILDER( Name("MyMatmul")
			 .Device(DEVICE_CPU)
			 .TypeConstraint<float>("T"), 
			 MyMatmulOp<CPUDevice, float, false>);

// Register for double
REGISTER_KERNEL_BUILDER( Name("MyMatmul")
			 .Device(DEVICE_CPU)
			 .TypeConstraint<double>("T"), 
			 MyMatmulOp<CPUDevice, double, false>);

// GPU
// Register for float
REGISTER_KERNEL_BUILDER( Name("MyMatmul")
			 .Device(DEVICE_GPU)
			 .TypeConstraint<float>("T"), 
			 MyMatmulOp<GPUDevice, float, false>);

// Register for double
REGISTER_KERNEL_BUILDER( Name("MyMatmul")
			 .Device(DEVICE_GPU)
			 .TypeConstraint<double>("T"), 
			 MyMatmulOp<GPUDevice, double, false>);



} // namespace tensorflow
