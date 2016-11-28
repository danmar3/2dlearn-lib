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
//   Description:   Test of element-wise pattern implementation
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
#include "twodlearn/common/cuda/elementwise_pattern_cu.h"

#define BLOCK_SIZE 1024

/* Includes, eigen */
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

/* Main */
int main(int argc, char **argv){   

  unsigned n= 4000;
  struct timespec start_cpu, end_cpu;
  
  // Allocate and fill h_A and h_B with data:
  TwinMat<double> a(n, n);
  a.transfer_h2d();
  
  TwinMat<double> b(n, n);
  b.transfer_h2d();
  
  TwinMat<double> c(n, n);
  //c.transfer_h2d();
  
  // For performance measure
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop); 
  
  // 1. --------------------------- power test ---------------------------
  // 1.1. power on cpu
  cout << "Running element-wise power" << endl;

  clock_gettime(CLOCK_MONOTONIC_RAW, &start_cpu);
  MatrixXd c_eigen = a.mat.array().pow(2.0);  
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_cpu);
  uint64_t cpu_time_ms = (1000000000L * (end_cpu.tv_sec - start_cpu.tv_sec) + 
  end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e6;
  
  // 1.2. power on gpu
  cout << "Running matmul on GPU" << endl;
  PowFunc<double> pow_cu(2.0);
  int n_blocks= 1 + ((a.mat.size() -1)/BLOCK_SIZE);
  dim3 dim_grid(n_blocks,1,1);
  dim3 dim_block(BLOCK_SIZE,1,1);
  cout << n_blocks << " " << BLOCK_SIZE << endl;
  
  cudaEventRecord(start);
  
  elementwise_pattern_cuda<PowFunc<double>, double, BLOCK_SIZE> <<<dim_grid, dim_block>>>(c.device, a.device, a.mat.size(), pow_cu);
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
  
}
