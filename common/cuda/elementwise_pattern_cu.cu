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
//   Description:   GPU implementation of element-wise pattern operations
//
//   ***********************************************************************


#include "elementwise_pattern_cu.h"

// look at http://docs.nvidia.com/cuda/thrust/#axzz4KT0tW3IU to see examples of how to use thurst


template <typename FunctorOp, typename T, int BLOCK_SIZE> 
__global__ void elementwise_pattern_cuda(T* in_tensor, int length, FunctorOp elementwise_op){
  // Block index
  int bx = blockIdx.x;
  
  // Thread index
  int tx = threadIdx.x;
  
  // create functor
  //FunctorOp elementwise_op;
  
  // get element index
  int a_idx = bx*BLOCK_SIZE + tx;

  // apply functor
  if (a_idx<length)
    in_tensor[a_idx] = elementwise_op( in_tensor[a_idx] );
}



template <typename FunctorOp, typename T, int BLOCK_SIZE> 
__global__ void elementwise_pattern_cuda(T* out_tensor, T* in_tensor, int length, FunctorOp elementwise_op){
  
  // Block index
  int bx = blockIdx.x;
  
  // Thread index
  int tx = threadIdx.x;
  
  // create functor
  //FunctorOp elementwise_op;
  
  // get element index
  int a_idx = bx*BLOCK_SIZE + tx;

  // apply functor
  if (a_idx<length)
    out_tensor[a_idx] = elementwise_op( in_tensor[a_idx] );
  
}


template <typename FunctorOp, typename T, int BLOCK_SIZE> 
__global__ void elementwise_pattern_cuda(T* out_tensor, T* in_tensor1, T* in_tensor2, int length, FunctorOp elementwise_op){
  // Block index
  int bx = blockIdx.x;
  
  // Thread index
  int tx = threadIdx.x;
  
  // create functor
  //FunctorOp elementwise_op;
  
  // get element index
  int a_idx = bx*BLOCK_SIZE + tx;

  // apply functor
  if (a_idx<length)
    out_tensor[a_idx] = elementwise_op( in_tensor1[a_idx], in_tensor2[a_idx] );
}
