/*

for mapping see: http://dovgalecs.com/blog/eigen-how-to-get-in-and-out-data-from-eigen-matrix/

Suppose you have an array with double values of size nRows x nCols.

    double *X; // non-NULL pointer to some data

You can create an nRows x nCols size double matrix using the Map functionality like this:

    MatrixXd eigenX = Map<MatrixXd>( X, nRows, nCols );

Now what if you have got an Eigen matrix with some result and you want to get out a plain double array? Guess what, you can use the Map once again!

    MatrixXd resultEigen;   // Eigen matrix with some result (non NULL!)

    double *resultC;                // NULL pointer

    Map<MatrixXd>( resultC, resultEigen.rows(), resultEigen.cols() ) = resultEigen;

*/


/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <functional>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "twodlearn/common/cuda/twin_object.h"

/* Includes, eigen */
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

/* Main */
int main(int argc, char **argv){   
  
  unsigned n= 4000;
  struct timespec start_cpu, end_cpu;

  // 1. Allocate and fill h_A and h_B with data:
  MatrixXd a_mat = MatrixXd::Random(n,n);
  double* (MatrixXd::*fn)(void) = &MatrixXd::data;
  //std::function<double*(void)> aux_func = std::bind(fn, &a_mat);, this works
  TwinObject<MatrixXd, double> a_mat_(&a_mat, a_mat.size()*sizeof(double), std::bind(fn, &a_mat));
  a_mat_.transfer_h2d();
  
  MatrixXd b_mat = MatrixXd::Random(n,n);
  TwinObject<MatrixXd, double> b_mat_(&b_mat, b_mat.size()*sizeof(double), b_mat.data());
  b_mat_.transfer_h2d();
  
  MatrixXd c_mat = MatrixXd::Random(n,n);
  TwinObject<MatrixXd, double> c_mat_(&c_mat, c_mat.size()*sizeof(double), c_mat.data());
  

  // 2. run matrix multiplication on CPU
  cout << "Running serial matmul" << endl;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start_cpu);

  MatrixXd c_eigen = a_mat * b_mat;
  
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
  cublasDgemm(c_handle, CUBLAS_OP_N, CUBLAS_OP_N, a_mat.rows(), b_mat.cols(), a_mat.cols(), 
	      &alpha, a_mat_.device, a_mat.rows(), b_mat_.device, b_mat.rows(), &beta, c_mat_.device, a_mat.rows());
  
  cudaEventRecord(stop);
  
  // transfer memory to host
  c_mat_.transfer_d2h();
  
  // 4. calculate the difference between the results in both implementations
  MatrixXd diff= c_eigen - c_mat;
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
