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

#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include "gmm_model_op.h"

#define BLOCK_SIZE 32

namespace tensorflow{

  namespace functor {
  
  // Definition of the functor for each device
  
  // when the functor is defined, all T are replaced by the given "class"
  // dev specifies the device where the operation will be executed
  template <typename T>
  struct GmmModelFunctor<CPUDevice, T> {
    void operator()( OpKernelContext* context,
		     Tensor& p_x_tf,
		     Tensor& gaussians_tf,
		     Tensor& sigma_inv_x_mu_tf,
		     const Tensor& x_tf,
		     const Tensor& w_tf,
		     const Tensor& mu_tf,
		     const Tensor& sigma_tf){
      
      const CPUDevice d = context->eigen_device<CPUDevice>(); // get device
      //Eigen::ThreadPool tp_interface(20);
      //Eigen::ThreadPoolDevice d(&tp_interface, 20);
      // Get inputs as eigen tensors
      auto x     = x_tf.tensor<T, 2>();
      auto w     = w_tf.tensor<T, 2>();  
      auto mu    = mu_tf.tensor<T, 3>();  
      auto sigma = sigma_tf.tensor<T, 3>();  
      // Get outputs as eigen tensors
      auto p_x   = p_x_tf.tensor<T, 1>();
      auto gaussians = gaussians_tf.tensor<T, 2>();
      auto sigma_inv_x_mu = sigma_inv_x_mu_tf.tensor<T, 3>();
      
      // Get number of samples, kernels and dimensions
      int n_samples = x_tf.dim_size(0);
      int n_params = mu_tf.dim_size(0);
      int n_kernels = mu_tf.dim_size(1);
      int n_dims = mu_tf.dim_size(2);

      // allocate memory for intermediate computations
      Tensor norm_const_tf;
      Tensor x_mu_tf;
      
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(), 
						     TensorShape({n_params, n_kernels}), 
						     &norm_const_tf));      
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(), 
						     TensorShape({n_samples, n_kernels, n_dims}), 
						     &x_mu_tf));
      
      auto norm_const = norm_const_tf.tensor<T, 2>();
      auto x_mu = x_mu_tf.tensor<T, 3>();
      
      // Coefficients
      Eigen::array<int, 1> red_dims1({1});
      Eigen::array<int, 1> red_dims2({2});
      //auto sigma_prod =  sigma.prod(red_dims).inverse(); // works
      //T pi_const = pow(2*PI, n_dims);
      norm_const.device(d) = pow(2*PI, n_dims)*(sigma.prod(red_dims2));
      norm_const.device(d) = norm_const.sqrt().inverse();
      
      // signa_inv
      auto sigma_inv = sigma.inverse();

      // x_mu     
      Eigen::array<int, 3> x_dims({n_samples, 1, n_dims});
      Eigen::array<int, 3> x_bcast({1, n_kernels, 1});
      
      if(n_params == n_samples){
	x_mu.device(d) = (x.reshape(x_dims)).broadcast(x_bcast) - mu;
	// sigma_inv_x_mu
	sigma_inv_x_mu.device(d) = x_mu * sigma_inv;

	// gaussians
	gaussians.device(d) = norm_const * (-0.5 * (x_mu*sigma_inv_x_mu).sum(red_dims2) ).exp();

	// final output
	p_x.device(d) = (w*gaussians).sum(red_dims1);
	
      }else if(n_params == 1) {
	//std::cout << "n_params: "<< n_params << ", n_samples: " << n_samples << std::endl;
	Eigen::array<int, 3> mu_bcast({n_samples, 1, 1});
	
	x_mu.device(d) = (x.reshape(x_dims)).broadcast(x_bcast) - mu.broadcast(mu_bcast);
	// sigma_inv_x_mu
	sigma_inv_x_mu.device(d) = x_mu * sigma_inv.broadcast(mu_bcast);

	// gaussians
	Eigen::array<int, 2> norm_bcast({n_samples, 1});
	gaussians.device(d) = norm_const.broadcast(norm_bcast) * (-0.5 * (x_mu*sigma_inv_x_mu).sum(red_dims2) ).exp();

	// final output
	p_x.device(d) = (w.broadcast(norm_bcast) * gaussians).sum(red_dims1);
      }
      
    }
    
  };
    
    
  
  // GPU implementation
  template <typename T>
  struct GmmModelFunctor<GPUDevice, T> {
    void operator()( OpKernelContext* context,
		     Tensor& p_x_tf,
		     Tensor& gaussians_tf,
		     Tensor& sigma_inv_x_mu_tf,
		     const Tensor& x_tf,
		     const Tensor& w_tf,
		     const Tensor& mu_tf,
		     const Tensor& sigma_tf){
      
      const GPUDevice d = context->eigen_device<GPUDevice>(); // get device
      //auto* stream = context->op_device_context()->stream();
      //OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
      //Eigen::GpuDevice d(stream) ; // get device
      
      // Get inputs as eigen tensors
      auto x     = x_tf.tensor<T, 2>();
      auto w     = w_tf.tensor<T, 2>();  
      auto mu    = mu_tf.tensor<T, 3>();  
      auto sigma = sigma_tf.tensor<T, 3>();  
      // Get outputs as eigen tensors
      auto p_x   = p_x_tf.tensor<T, 1>();
      auto gaussians = gaussians_tf.tensor<T, 2>();
      auto sigma_inv_x_mu = sigma_inv_x_mu_tf.tensor<T, 3>();
      
      // Get number of samples, kernels and dimensions
      int n_samples = x_tf.dim_size(0);
      int n_params = mu_tf.dim_size(0);
      int n_kernels = mu_tf.dim_size(1);
      int n_dims = mu_tf.dim_size(2);

      // allocate memory for intermediate computations
      Tensor norm_const_tf;
      Tensor x_mu_tf;
      
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(), 
						     TensorShape({n_params, n_kernels}), 
						     &norm_const_tf));      
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(), 
						     TensorShape({n_samples, n_kernels, n_dims}), 
						     &x_mu_tf));
      
      auto norm_const = norm_const_tf.tensor<T, 2>();
      auto x_mu = x_mu_tf.tensor<T, 3>();
      
      // Coefficients
      Eigen::array<int, 1> red_dims1({1});
      Eigen::array<int, 1> red_dims2({2});
      //auto sigma_prod =  sigma.prod(red_dims).inverse(); // works
      //T pi_const = pow(2*PI, n_dims);
      norm_const.device(d) = pow(2*PI, n_dims)*(sigma.prod(red_dims2));
      norm_const.device(d) = norm_const.sqrt().inverse();
      
      // signa_inv
      auto sigma_inv = sigma.inverse();

      // x_mu     
      Eigen::array<int, 3> x_dims({n_samples, 1, n_dims});
      Eigen::array<int, 3> x_bcast({1, n_kernels, 1});
      
      if(n_params == n_samples){
	//std::cout << "okkkk"<< std::endl;
	x_mu.device(d) = (x.reshape(x_dims)).broadcast(x_bcast) - mu;
	//x_mu.device(d) = mu;
	// sigma_inv_x_mu
	sigma_inv_x_mu.device(d) = x_mu * sigma_inv;
	x_mu.device(d) = x_mu * sigma_inv_x_mu;
	// gaussians
	gaussians.device(d) = w * norm_const * (-0.5 * x_mu.sum(red_dims2) ).exp();
	//gaussians.device(d) = w*gaussians;
	// final output
	p_x.device(d) = gaussians.sum(red_dims1);
	gaussians.device(d) = gaussians/w;
	
      }else if(n_params == 1) {
	//std::cout << "n_params: "<< n_params << ", n_samples: " << n_samples << std::endl;
	Eigen::array<int, 3> mu_bcast({n_samples, 1, 1});
	
	x_mu.device(d) = (x.reshape(x_dims)).broadcast(x_bcast) - mu.broadcast(mu_bcast);
	// sigma_inv_x_mu
	sigma_inv_x_mu.device(d) = x_mu * sigma_inv.broadcast(mu_bcast);
	x_mu.device(d) = x_mu * sigma_inv_x_mu;
	// gaussians
	Eigen::array<int, 2> norm_bcast({n_samples, 1});
	gaussians.device(d) = x_mu.sum(red_dims2);
	gaussians.device(d) = (-0.5*gaussians).exp();
	gaussians.device(d) = norm_const.broadcast(norm_bcast) * gaussians;
	//gaussians.device(d) = norm_const.broadcast(norm_bcast) * (-0.5 * (x_mu).sum(red_dims2) ).exp();
	
	// final output
	gaussians.device(d) = w.broadcast(norm_bcast) * gaussians;
	p_x.device(d) = (gaussians).sum(red_dims1);
	gaussians.device(d) = gaussians / w.broadcast(norm_bcast);
	
      }
      
    }
  };
    
} // end namespace functor
  



// Device: DEVICE_CPU, DEVICE_GPU
// T: type of the matrix elements: float, or double
// USE_CUBLAS: true for gpu, false for cpu
template <typename Device, typename T>
class GmmModelOp : public OpKernel {
public:
  explicit GmmModelOp(OpKernelConstruction* context) : OpKernel(context) {}
  
  void Compute(OpKernelContext* context) override {
    // get input tensors
    const Tensor& x = context->input(0);
    const Tensor& w = context->input(1);
    const Tensor& mu = context->input(2);
    const Tensor& sigma = context->input(3);

    // check tensors have correct dimensions
    OP_REQUIRES(context, x.dims() == 2,
                errors::InvalidArgument("x must be 2 dimensional"));
    OP_REQUIRES(context, w.dims() == 2,
		errors::InvalidArgument("w must be 2 dimensional"));
    OP_REQUIRES(context, mu.dims() == 3,
		errors::InvalidArgument("mu must be 3 dimensional"));
    OP_REQUIRES(context, sigma.dims() == 3,
		errors::InvalidArgument("sigma must be 3 dimensional"));
    
    // check dimentions are valid
    OP_REQUIRES(context, x.dim_size(1) == mu.dim_size(2),
                errors::InvalidArgument("dimensions of arguments x and mu do not agree"));
    OP_REQUIRES(context, x.dim_size(1) == sigma.dim_size(2),
                errors::InvalidArgument("dimensions of arguments x and sigma do not agree"));

    // obtain the dimentions for the output matrix
    TensorShape p_x_shape({x.dim_size(0)});
    TensorShape gaussians_shape({x.dim_size(0), mu.dim_size(1)});
    TensorShape aux2_shape({x.dim_size(0), mu.dim_size(1), mu.dim_size(2)});

    // Allocate the output tensor, this tells tensorflow
    // that output corresponds to the output id 0
    Tensor* p_x = nullptr;
    Tensor* gaussians = nullptr;
    Tensor* aux2 = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, p_x_shape, &p_x));
    OP_REQUIRES_OK(context, context->allocate_output(1, gaussians_shape, &gaussians));
    OP_REQUIRES_OK(context, context->allocate_output(2, aux2_shape, &aux2));
    
    
    // call the function 
    functor::GmmModelFunctor<Device, T> functor_launch;
    functor_launch(context, *p_x, *gaussians, *aux2, x, w, mu, sigma);
    
  }
};


// Register the operations
REGISTER_OP("GmmModel")
.Input("x: T")
.Input("w: T")
.Input("mu: T")
.Input("sigma: T")
.Output("p_x: T")
.Output("gaussians: T")
.Output("aux2: T")
.Attr("T: {float, double}")
.Doc(R"doc(Computes the probability for a set of samples ussing a gaussian mixture model = sum_k (w_k * Gaussian(x_i, mu_k, sigma_k)) )doc");

// Register for float
REGISTER_KERNEL_BUILDER( Name("GmmModel")
			 .Device(DEVICE_CPU)
			 .TypeConstraint<float>("T"), 
			 GmmModelOp<CPUDevice, float>);

// Register for double
REGISTER_KERNEL_BUILDER( Name("GmmModel")
			 .Device(DEVICE_CPU)
			 .TypeConstraint<double>("T"), 
			 GmmModelOp<CPUDevice, double>);

// GPU
// Register for float
REGISTER_KERNEL_BUILDER( Name("GmmModel")
			 .Device(DEVICE_GPU)
			 .TypeConstraint<float>("T"), 
			 GmmModelOp<GPUDevice, float>);

// Register for double
REGISTER_KERNEL_BUILDER( Name("GmmModel")
			 .Device(DEVICE_GPU)
			 .TypeConstraint<double>("T"), 
			 GmmModelOp<GPUDevice, double>);

			 

} // namespace tensorflow
