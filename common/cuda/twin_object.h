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
//   Description:   Class that handles the memory storage of a given object in host and device memory
//
//   ***********************************************************************


#ifndef TWIN_OBJECT_H_
#define TWIN_OBJECT_H_

#include <stdlib.h>
#include <cuda.h>
#include <functional>
#include "cuda_error.h"

template <class Obj, typename T>
class TwinObject{
 public:
  Obj* obj_ptr; // pointer to the host object
  
  T* host;   // pointer to host memory
  T* device;   // pointer to device memory
  
  std::size_t obj_size; // size of the object
  
  std::function<T*(void)> get_h;
  
  /* Methods */
  // constructors
 TwinObject( ) : obj_ptr(nullptr),
    obj_size(0),
    host(nullptr),
    device(nullptr) {}

 TwinObject(Obj* in_obj, std::size_t in_size, T* h_in )
   : obj_ptr(in_obj), 
    obj_size(in_size),
    host(h_in) {
      // allocate memory on device
      CUDA_CHECK( cudaMalloc((void **) &device, obj_size) );
  }
  
 TwinObject(Obj* in_obj, std::size_t in_size, std::function<T*(void)> get_h_in)
   : obj_ptr(in_obj), 
    obj_size(in_size),
    get_h(get_h_in) {
      // allocate memory on device
      CUDA_CHECK( cudaMalloc((void **) &device, obj_size) );
  }
  
  // destructor
  ~TwinObject(){
    // free cuda memory
    cudaFree(device);
  }

  void initialize(Obj* in_obj, std::size_t in_size, std::function<T*(void)> get_h_in) {
    obj_ptr = in_obj;
    obj_size = in_size;
    get_h = get_h_in;
    
    // allocate memory on device
    CUDA_CHECK( cudaMalloc((void **) &device, obj_size) );
  }
  
  
  void set_get_h(std::function<T*(void)> get_h_in){
    get_h = get_h_in;
  }
  
  
  void transfer_d2h(){
    if (get_h != nullptr)
      host = get_h();
    
    CUDA_CHECK( cudaMemcpy(host, device, obj_size, cudaMemcpyDeviceToHost) );
  }
  
  void transfer_h2d(){
    if (get_h != nullptr){
      host = get_h();
    }
    
    CUDA_CHECK( cudaMemcpy(device, host, obj_size, cudaMemcpyHostToDevice) );
  }
  
  void free_device(){
    cudaFree(device);
  }
  
  void realocate_device(){
    CUDA_CHECK( cudaMalloc((void **) &device, obj_size) );
  }
};


#endif // 
