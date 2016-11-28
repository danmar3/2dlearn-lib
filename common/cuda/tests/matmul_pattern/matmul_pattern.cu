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
//   Description:   Test of matmul pattern implementation
//
//   ***********************************************************************


/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <functional>

/* Includes, cuda */
// #include <cuda_runtime.h>
#include "twodlearn/common/cuda/eigen_cuda.h"
#include "twodlearn/common/cuda/matmul_pattern_cu.h"

#define BLOCK_SIZE 48

/* Includes, eigen */
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

/* Main */
int main(int argc, char **argv){   

  unsigned m= 4000;
  unsigned k= 4000;
  unsigned n= 4000;
  
  struct timespec start_cpu, end_cpu;
  
  // Allocate and fill h_A and h_B with data:
  TwinMat<double, RowMajor> a(m, k);
  a.transfer_h2d();
  
  TwinMat<double, RowMajor> b(k, n);
  b.transfer_h2d();
  
  TwinMat<double, RowMajor> c(m, n);
  //c.transfer_h2d();
  
  // For performance measure
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop); 
  
  // 1. --------------------------- matmul test ---------------------------
  // 1.1. matmul on cpu
  cout << "Running matmul on cpu" << endl;
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &start_cpu);
  MatrixXd c_eigen = a.mat * b.mat;  
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_cpu);
  uint64_t cpu_time_ms = (1000000000L * (end_cpu.tv_sec - start_cpu.tv_sec) + 
  end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e6;
  
  // 1.2. matmul on gpu
  cout << "Running matmul on GPU" << endl;
  MulFunc<double> mul_cu;
  SumFunc<double> sum_cu;
  dim3 dim_grid( 1 + ((c.mat.cols() -1)/BLOCK_SIZE),
		 1 + ((c.mat.rows() -1)/BLOCK_SIZE),
		 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE,1);
  cout << dim_grid.x << " " << dim_grid.y << " " << BLOCK_SIZE << endl;
  
  cudaEventRecord(start);
  
  matmul_pattern_cuda<MulFunc<double>, SumFunc<double>, double, BLOCK_SIZE> <<<dim_grid, dim_block>>>(c.device, a.device, b.device, a.mat.rows(), a.mat.cols(), b.mat.cols(), mul_cu, sum_cu);
  cudaDeviceSynchronize();
  c.transfer_d2h();  
  
  cudaEventRecord(stop);
  
  // 1.3. print performance
  // error
  MatrixXd diff= c_eigen - c.mat;
  diff = diff.array().pow(2);
  cout << "difference: " << diff.sum() << "\n";
  // time
  cudaEventSynchronize(stop);
  float gpu_time_ms;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);
  cout << "time on cpu: " << cpu_time_ms << "[ms] \n";
  cout << "time on gpu: " << gpu_time_ms << "[ms] \n";

  /*
  cout << "a: \n";
  cout << a.mat <<"\n";
  cout << "b: \n";
  cout << b.mat <<"\n";
  cout << "c_gpu: \n";
  cout << c.mat <<"\n";
  cout << "c_cpu: \n";
  cout << c_eigen <<"\n";
  
  cout << "time on gpu: " << gpu_time_ms << "[ms] \n";
  */

  /*
  // 2. ------------------------ element-wise matmul ---------------------------
  // 2.1. element-wise matmul on cpu
  cout << "\n\nRunning element-wise multiplication" << endl;

  clock_gettime(CLOCK_MONOTONIC_RAW, &start_cpu);
  c_eigen = a.mat.cwiseProduct(b.mat); ; // a.array() * n.array(); 
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_cpu);
  cpu_time_ms = (1000000000L * (end_cpu.tv_sec - start_cpu.tv_sec) + 
  end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e6;
  
  // 2.2. element-wise matmul on gpu
  cout << "Running matmul on GPU" << endl;
  EWMatmulFunc<double> ew_matmul_cu(1.0);
  //n_blocks= 1 + ((a.mat.size() -1)/BLOCK_SIZE);
  //dim_grid(n_blocks,1,1);
  //dim_block(BLOCK_SIZE,1,1);
  cout << n_blocks << " " << BLOCK_SIZE << endl;
  
  cudaEventRecord(start);
  
  elementwise_pattern_cuda<EWMatmulFunc<double>, double, BLOCK_SIZE> <<<dim_grid, dim_block>>>(c.device, a.device, b.device, a.mat.size(), ew_matmul_cu);
  cudaDeviceSynchronize();
  c.transfer_d2h();  
  
  cudaEventRecord(stop);
  
  // 2.3. print performance
  // error
  diff= c_eigen - c.mat;
  diff = diff.array().pow(2);
  cout << "difference: " << diff.sum() << "\n";
  // time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time_ms, start, stop);
  cout << "time on cpu: " << cpu_time_ms << "[ms] \n";
  cout << "time on gpu: " << gpu_time_ms << "[ms] \n";
  */
}
