# 2D-learn
2D-learn is a library for machine learning research. 

## tf-lib:
classes and functions for being used on tensorflow
 
## common:

 - cuda: GPU implementation of common parallel patterns

## installation:

 1. required packages for tf-lib: Anaconda 3, TensorFlow, CUDA 7.0, Eigen 3.2
    
 2. clone the repository: <br>    
    mkdir ~/libraries <br>    
    cd ~/libraries <br>    
    git clone https://github.com/danmar3/2dlearn-lib.git twodlearn

 3. add the following environmental variables to your bashrc:  
    export CUDA_HOME="path to cuda toolkit"  
    export EIGEN_HOME="path to eigen installation"  
    export TWODLEARN_HOME="/home/username/libraries/"  
    
 4. add TWODLEARN_HOME to the python path: <br>
    export PYTHONPATH="$TWODLEARN_HOME:$PYTHONPATH" 

 6. Compile the tf-lib operations (Not needed for mat-lib) <br>
    cd tf_lib/ops/kernels <br>
    make

 
## Developers:

Daniel L. Marino (marinodl@vcu.edu)

Modern Heuristics Research Group (MHRG) <br>
Virginia Commonwealth University (VCU), Richmond, VA <br>
http://www.people.vcu.edu/~mmanic/