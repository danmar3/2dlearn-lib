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
//   Description:   GPU implementation of matrix multiplication pattern operation
//
//   ***********************************************************************


#ifndef MATMUL_PATTERN_CU_H_
#define MATMUL_PATTERN_CU_H_

#include <cuda.h>



/* device functors definition*/
template<typename T>
struct CMulFunc{
  const T a;
  
  CMulFunc(T _a) : a(_a){}
  
  __host__ __device__ 
  T operator() (const T& x1, const T& x2){
    return a * x1 * x2;
  }
};

template<typename T>
struct MulFunc{  
  
  __host__ __device__ 
  T operator() (const T& x1, const T& x2) {
    return x1 * x2;
  }
};

template<typename T>
struct SumFunc{
  
  __host__ __device__ 
  T operator() (const T& x1, const T& x2){
    return x1 + x2;
  }
};

template<typename T>
struct SquaredDiffFunc{
  
  __host__ __device__ 
  T operator() (const T& x1, const T& x2){
    return (x1 - x2)*(x1 - x2);
  }
};


/* kernels definition */
template <typename FunctorOp1, typename FunctorOp2, typename T, int BLOCK_SIZE> 
  __global__ void matmul_pattern_cuda(T* out_mat, T* in_mat1, T* in_mat2, int m, int k, int n, FunctorOp1 elementwise_op, FunctorOp2 reduction_op);




/* ----------------------------- kernel implementation -------------------------------- */

template <typename FunctorOp1, typename FunctorOp2, typename T, int BLOCK_SIZE> 
  __global__ void matmul_pattern_cuda(T* c_mat, const T* a_mat, const T* b_mat, int m, int k, int n, FunctorOp1 elementwise_op, FunctorOp2 reduction_op){
  
  // define shared memory
  __shared__ T a_mat_s[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ T b_mat_s[BLOCK_SIZE][BLOCK_SIZE];

  // Block index
  int by = blockIdx.y; int bx = blockIdx.x;  
  
  // Thread index
  int ty = threadIdx.y; int tx = threadIdx.x;
  
  // output index for the thread
  int row = by * BLOCK_SIZE + ty ;
  int col = bx * BLOCK_SIZE + tx ;
  
  // output
  T c_out = 0.0;
  
  
  for (int bk_idx=0; bk_idx < (1 + (k - 1)/BLOCK_SIZE); bk_idx++ ) { // bk_idx: block k index
    // load values of matrix a_mat
    if ( (row < m) && (bk_idx*BLOCK_SIZE + tx < k)  ){
      // get a_mat[row][bk_idx * BLOCK_SIZE + tx]
      a_mat_s[ty][tx] = a_mat[ row*k + bk_idx * BLOCK_SIZE + tx ]; 
    } else {
      a_mat_s[ty][tx] = 0.0;
    }
    
    // load values of matrix b_mat
    if ( (bk_idx*BLOCK_SIZE + ty < k) && (col < n)  ){
      // get b_mat[bk_idx * BLOCK_SIZE + ty][col]
      b_mat_s[ty][tx] = b_mat[ (bk_idx*BLOCK_SIZE + ty)*n + col ]; 
    } else {
      b_mat_s[ty][tx] = 0.0;
    }
    
    __syncthreads();
    
    // calculate the matrix multiplication for the elements 
    // in shared memory
    if (row<m && col<n) {
#pragma unroll
      for(int k_idx=0; k_idx < BLOCK_SIZE; k_idx++)
	//c_out += a_mat_s[ty][k_idx] * b_mat_s[k_idx][tx]; // standard matrix multiplication
	// TODO: add reduction operation, in this case is just the sum
	c_out = reduction_op(c_out, elementwise_op(a_mat_s[ty][k_idx], b_mat_s[k_idx][tx]));
    }
    
    __syncthreads();
  }
  
  if (row<m && col<n)
    c_mat [row*n + col] = c_out; 
}
  

  
    

#endif
