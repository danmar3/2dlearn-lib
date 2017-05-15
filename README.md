# 2D-learn
2D-learn is a library for machine learning research. The library has two main components: 1) mat-lib, a library for Matlab for numerical optimization; 2) tf-lib, a library for TensorFlow that defines common machine learning models.

## mat-lib:
A Matlab library inspired by TensorFLow and Torch for numerical optimization using backpropagation. The library defines composite functions using computation graphs, where backpropagation can be used to perform numerical optimization.

## tf-lib:
classes and functions for being used on tensorflow
 
## common:

 - cuda: GPU implementation of common parallel patterns

## installation:

 1. required packages for tf-lib: Anaconda 3, TensorFlow, CUDA 7.0, Eigen 3.2
    
 2. clone the repository: 
    
    mkdir ~/libraries
    
    cd ~/libraries
    
    git clone https://github.com/danmar3/2dlearn-lib.git twodlearn

 3. add the following environmental variables to your bashrc:  
    export CUDA_HOME="path to cuda toolkit"  
    export EIGEN_HOME="path to eigen installation"  
    export TWODLEARN_HOME="/home/username/libraries/"  
    
 4. add TWODLEARN_HOME to the python path: <br>
    export PYTHONPATH="$TWODLEARN_HOME:$PYTHONPATH"

 6. Compile the tf-lib operations (Not needed for mat-lib)
    cd tf_lib/ops/kernels
    make

 5. add the twodlearn/mat-lib path to the start-up Matlab script:
    addpath('/home/username/libraries/twodlearn/mat_lib/')


## Developers:

Daniel L. Marino (marinodl@vcu.edu)

Modern Heuristics Research Group (MHRG)
Virginia Commonwealth University (VCU), Richmond, VA
http://www.people.vcu.edu/~mmanic/