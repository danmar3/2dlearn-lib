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
//   Description:   Test of matmul multiplication using cublas
//
//   ***********************************************************************



/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <functional>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "twodlearn/common/cuda/eigen_cuda.h"

/* Includes, eigen */
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

/* Main */
int main(int argc, char **argv){   
  
  unsigned n= 4000;
  struct timespec start_cpu, end_cpu;
  
  // 1. Allocate and fill h_A and h_B with data:
  TwinMat<double> a(n, n);
  a.transfer_h2d();
  
  TwinMat<double> b(n, n);
  b.transfer_h2d();
  
  TwinMat<double> c(n, n);
  c.transfer_h2d();

  // 2. run matrix multiplication on CPU
  cout << "Running serial matmul" << endl;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start_cpu);

  MatrixXd c_eigen = a.mat * b.mat;
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_cpu);
  uint64_t cpu_time_ms = (1000000000L * (end_cpu.tv_sec - start_cpu.tv_sec) + 
			  end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e6;
  
  // 3. Create cublas handle
  cublasHandle_t c_handle;
  cublasCreate(&c_handle);
  
  // define devices
  //const int nDevices = 2;
  //int deviceId[nDevices] = {0, 1};
  
  
  // For performance measure
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop); 

  // 3.1. run matrix multiplication on GPU
  cout << "Running matmul on GPU" << endl;
  
  double alpha = 1.0;
  double beta = 0.0;

  cudaEventRecord(start);
  
  cublasDgemm(c_handle, CUBLAS_OP_N, CUBLAS_OP_N, a.mat.rows(), b.mat.cols(), a.mat.cols(), 
	      &alpha, a.device, a.mat.rows(), b.device, b.mat.rows(), &beta, c.device, a.mat.rows());
  
  cudaEventRecord(stop);
  
  // transfer memory to host
  c.transfer_d2h();

  // 4. calculate the difference between the results in both implementations
  MatrixXd diff= c_eigen - c.mat;
  diff = diff.array().pow(2);
  cout << "difference: " << diff.sum() << "\n";
  
  // 5. show the time used by each operation
  cudaEventSynchronize(stop);
  float gpu_time_ms;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);
  cout << "time on cpu: " << cpu_time_ms << "[ms] \n";
  cout << "time on gpu: " << gpu_time_ms << "[ms] \n";
  // destroy cublasXt handle
  cublasDestroy(c_handle);

}
