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
//   Description:   Test implementing 1-nearest neighbor classifier
//
//   ***********************************************************************


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

/* Includes, cuda */
#include "twodlearn/common/cuda/eigen_cuda.h"
#include "twodlearn/common/cuda/matmul_pattern_cu.h"

/* Includes, eigen */
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

#define BLOCK_SIZE 12

typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdR;


void dataset2mat(ArffData* dataset, MatrixXdR& dataset_x, MatrixXdR& dataset_y){
  // allocate memory
  dataset_x = MatrixXdR::Zero(dataset->num_instances(), dataset->num_attributes());
  dataset_y = MatrixXdR::Zero(dataset->num_instances(), 1);
  // populate matrices
  for(int i = 0; i < dataset->num_instances(); i++){ 
    for(int j = 0; j < dataset->num_attributes() - 1; j++){ 
      dataset_x(i,j) = (double) dataset->get_instance(i)->get(j)->operator float();
    }
    dataset_y(i,0) = (double)
      dataset->get_instance(i)->get(dataset->num_attributes()-1)->operator float();
  }
}


int* gpu_knn( MatrixXdR& dataset_x, MatrixXdR& dataset_y){
  
  cudaEvent_t start_gpu, stop_gpu;                           // performance evaluation
  cudaEventCreate(&start_gpu); cudaEventCreate(&stop_gpu);   // performance evaluation


  int* predictions = (int*)malloc(dataset_x.rows() * sizeof(int));
  
  // calculate distance between all elements
  //MatrixXd dist = MatrixXd::Zero(dataset_x.rows(), dataset_x.rows());
  
  TwinMat<double, RowMajor> a;
  a = dataset_x;
  cout << a.mat.rows() << " " << a.mat.cols() << " "<< a.obj_size << endl;
  //a.transfer_h2d();
  
  TwinMat<double, RowMajor> b;
  MatrixXdR aux = dataset_x.transpose();
  b = aux;
  //b = dataset_x.transpose();
  //cout << b.mat.rows() << " " << b.mat.cols() << " "<< b.obj_size << endl;
  //b.transfer_h2d();
  
  
  TwinMat<double, RowMajor> dist((int)a.mat.rows(), (int)b.mat.cols());
  
  TwinMat<double, RowMajor> dist2((int)a.mat.rows(), (int)b.mat.cols());
  
  cudaEventRecord(start_gpu);                                // performance evaluation
  //dist.transfer_h2d();
  cout << "Matrices allocated in cpu and gpu" << endl;
  

  
 
  // calculate distance between all elements
  a.transfer_h2d();
  b.transfer_h2d();
  
  cout << "Calculating distance matrix on GPU" << endl;
  SquaredDiffFunc<double> dist_cu;
  MulFunc<double> mul_cu;
  SumFunc<double> sum_cu;
  dim3 dim_grid( 1 + ((dist.mat.rows() -1)/BLOCK_SIZE),
		 1 + ((dist.mat.cols() -1)/BLOCK_SIZE),
		 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE,1);
  cout << dim_grid.x << " " << dim_grid.y << " " << BLOCK_SIZE << endl;
      
  matmul_pattern_cuda<SquaredDiffFunc<double>, SumFunc<double>, double, BLOCK_SIZE> <<<dim_grid, dim_block>>>(dist.device, a.device, b.device, a.mat.rows(), a.mat.cols(), b.mat.cols(), dist_cu, sum_cu);
  
  cudaDeviceSynchronize();
  dist.transfer_d2h();  
  
  cudaEventRecord(stop_gpu);                                 // performance evaluation
  cudaEventSynchronize(stop_gpu);                            // performance evaluation
  float gpu_time_ms;                                         // performance evaluation
  cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);           // performance evaluation
  cout << "time on gpu: " << gpu_time_ms << "[ms] \n";       // performance evaluation

 
  
  
  // calculate index with minimum distance
  int min_idx;
  for(int i=0; i<dist.mat.rows(); i++){
    dist.mat(i,i)= 1e10;
    dist.mat.row(i).minCoeff(&min_idx);
    
    predictions[i] = dataset_y(min_idx,0);
    //cout << predictions[i] << endl;
  }
  
  return predictions;
}



int* KNN(ArffData* dataset){
  
  int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
  
  for(int i = 0; i < dataset->num_instances(); i++){ // for each instance in the dataset
    
    float smallestDistance = FLT_MAX;
    int smallestDistanceClass;
    
    for(int j = 0; j < dataset->num_instances(); j++){ // target each other instance        
      
      if(i == j) continue;
      
      float distance = 0;
      
      for(int k = 0; k < dataset->num_attributes() - 1; k++){ // compute the distance between the two instances           
	float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
	distance += diff * diff;
      }
      
      distance = sqrt(distance);
      
      if(distance < smallestDistance){ // select the closest one
	smallestDistance = distance;
	smallestDistanceClass = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32();
      }
    }
    
    predictions[i] = smallestDistanceClass;
  }
  
  return predictions;
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset){
  
  int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses
  
  for(int i = 0; i < dataset->num_instances(); i++){ // for each instance compare the true class and predicted class
    
    int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
    int predictedClass = predictions[i];
    
    confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
  }
  
  return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset){
  
  int successfulPredictions = 0;
  
  for(int i = 0; i < dataset->num_classes(); i++){
    successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagnoal are correct predictions
  }
  
  return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[]){
  if(argc < 2 or argc > 3 ){
    cout << "Usage: ./main datasets/datasetFile.arff num_threads" << endl;
    exit(0);
  }

  unsigned n_threads = 0;
  if(argc == 3)
    n_threads= atoi(argv[2]);
  
  ArffParser parser(argv[1]);
  ArffData *dataset = parser.parse();
  struct timespec start, end;
  
  cout << "Number of instances: " << dataset->num_instances() << "\n";
  cout << "Number of atributes: " << dataset->num_attributes() << "\n\n";
  
  // ----------------------- serial code ------------------------//
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  int* predictions = KNN(dataset);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  
  int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
  float accuracy = computeAccuracy(confusionMatrix, dataset);
  
  
  uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
  
  printf("The 1NN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);

  

  //----------------- GPU implementation ------------------------//
  MatrixXdR dataset_x, dataset_y;
  cout << "formating dataset ..." << endl;
  dataset2mat(dataset, dataset_x, dataset_y);
  cout << "formating Done" << endl;
  
  
  // For performance measure 
  cudaEvent_t start_gpu, stop_gpu;
  cudaEventCreate(&start_gpu); cudaEventCreate(&stop_gpu);
  
  // 3.1. run matrix multiplication on GPU 
  cout << "\n\nRunning on GPU" << endl;
  
  cudaEventRecord(start_gpu);
  int* predictions_gpu =  gpu_knn( dataset_x, dataset_y );
  cudaEventRecord(stop_gpu);
  
  // evaluate accuracy
  
  int* confusionMatrix_gpu = computeConfusionMatrix(predictions_gpu, dataset);
  float accuracy_gpu = computeAccuracy(confusionMatrix_gpu, dataset);
    
  float gpu_time_ms;
  cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu); 
  //cout << "time on gpu: " << gpu_time_ms << "[ms] \n";
  printf("The 1NN classifier for %lu instances required %f ms GPU time, accuracy was %.4f\n", dataset->num_instances(), gpu_time_ms, accuracy_gpu);
  
  
  
  
}
