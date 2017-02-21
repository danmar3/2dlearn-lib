# 2dlearn-lib
A collection of classes and functions used for machine learning

## tf-lib:
classes and functions for being used on tensorflow
 
## common:

 - cuda: GPU implementation of common parallel patterns

## installation:

 1. install required packages: Anaconda 3, TensorFlow, CUDA 7.0, Eigen 3.2
 2. clone the repository: 
    
    mkdir ~/libraries
    cd ~/libraries
    git clone https://github.com/danmar3/2dlearn-lib.git twodlearn

 3. add the following environmental variables to your bashrc:  
    export CUDA_HOME="path to cuda toolkit"  
    export EIGEN_HOME="path to eigen installation"  
    export TWODLEARN_HOME="/home/username/libraries/"  
    
 4. add TWODLEARN_HOME to the python path
    export PYTHONPATH="$TWODLEARN_HOME:$PYTHONPATH"
