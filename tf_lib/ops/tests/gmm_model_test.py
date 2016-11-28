#************************************************************************
#      __   __  _    _  _____   _____
#     /  | /  || |  | ||     \ /  ___|
#    /   |/   || |__| ||    _||  |  _
#   / /|   /| ||  __  || |\ \ |  |_| |
#  /_/ |_ / |_||_|  |_||_| \_\|______|
#    
# 
#   Written by: Daniel L. Marino (marinodl@vcu.edu) (2016)
#
#   Copyright (2016) Modern Heuristics Research Group (MHRG)
#   Virginia Commonwealth University (VCU), Richmond, VA
#   http://www.people.vcu.edu/~mmanic/
#   
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#  
#   Any opinions, findings, and conclusions or recommendations expressed 
#   in this material are those of the author's(s') and do not necessarily 
#   reflect the views of any other entity.
#  
#   ***********************************************************************
#
#   Description: Test of the Gaussian Mixture Model Implementation 
#
#   ***********************************************************************

import tensorflow as tf
import twodlearn.tf_lib.ops as tdl

import numpy as np
import collections
from twodlearn.tf_lib.datasets.generic import Datasets
from twodlearn.tf_lib.GMM import *

## 1. Create dataset
def gmm_sampling(mu, sigma, w, n_samples=1):
    # generates random vectors, sampled from a gausian mixture model
    #  
    #     - mu: 3d tensor containing the means for the gaussians.
    #           the "depth" dimention (3rd) is used to index the
    #           gaussians.    [ kernel_id, dim]
    #     - sigma: 3d tensor containing the covariance matrix of the
    #              gaussians. [ kernel_id, dim] for diagonal matrices
    #     - w: vector in form of a 3d tensor containing the weights
    #          for each one of the gaussians, they have to sum one. 
    #          [kernel_id]
    n_kernels = mu.shape[0]
    n_dim = mu.shape[1]
    
    # random sample the kernel from which the output samples are going
    # to be drawn
    kernel_id= np.argmax(np.random.multinomial(1, w, size=[n_samples]), axis=1 )
    
    out = np.zeros([n_samples, n_dim])
    for i in range(n_samples):
        out[i,:]= np.random.multivariate_normal(mu[kernel_id[i],:], np.diag(sigma[kernel_id[i],:])) # if diagonal
    
    return out;

n_samples = 1000

n_kernels_r = 2
n_dim = 2
mu_r = np.array([[1, 1], [10, 10]])
sigma_r = np.array([[0.5, 0.5], [2, 2]])
w_r = [1/n_kernels_r]*n_kernels_r # has to sum to one

# random sample from reference distribution
aux_x = gmm_sampling(mu_r, sigma_r, w_r, n_samples);

# build the dataset
data = Datasets(aux_x)
data.normalize()
print('Data shape: ', data.train.x.shape)

## 2. Create model
with tf.Session('') as sess:
    n_kernels = 2

    _ref_model = GmmShallowModel( n_dim, n_kernels, diagonal= True)
    ref_model = _ref_model.setup(n_samples)
    
    print("ref_model.out: ", ref_model.out.get_shape())
    print("ref_model.w: ", ref_model.w.get_shape())
    print("ref_model.mu: ", ref_model.mu.get_shape())
    print("ref_model.sigma: ", ref_model.sigma.get_shape())
    
    gmm_test, _gaussians, _aux2 = tdl.gmm_model(ref_model.inputs, ref_model.w, ref_model.mu, ref_model.sigma)
    
    error_tf = tf.nn.l2_loss( ref_model.out - gmm_test)
    
    tf.initialize_all_variables().run()
    
    print("\n\n --------------- running -----------------\n\n")
    
    [error, l_ref, p_x_ref, p_x_test] = sess.run([error_tf, ref_model.loss, ref_model.out, gmm_test],feed_dict={ref_model.inputs : data.train.x})
    
    print("error: ", error)
    print("p_x_test: ", p_x_test.shape)
    print(p_x_test[0:10])
    
    
    print("\n\n --------------- gradient check -----------------\n\n")
    n_samples = 50;
    x_grad = tf.Variable (tf.truncated_normal( [n_samples, n_dim], stddev= 0.1 ))
    
    gmm_test2, _gaussians, _aux2 = tdl.gmm_model(x_grad, ref_model.w, ref_model.mu, tf.exp(ref_model.sigma))
    grad_target = tf.reduce_sum(gmm_test2)
    
    tf.initialize_all_variables().run()
    print("ref_model.sigma:", ref_model.sigma.get_shape())
    print("ref_model.inputs:", ref_model.inputs.get_shape())
    #gradient_error = tf.test.compute_gradient_error( [a_mat, b_mat], [a_mat.get_shape(), b_mat.get_shape()] ,loss, [1])
    
    for i in range(100):
        '''
        gradient_error = tf.test.compute_gradient_error( [x_grad, ref_model.w, ref_model.mu, ref_model.sigma],
                                                        [(n_samples, n_dim), 
                                                         (1, n_kernels), 
                                                         (1, n_kernels, n_dim), 
                                                         (1, n_kernels, n_dim)] , grad_target, [1])
                                                         
        '''
        #gradient_error = tf.test.compute_gradient_error( x_grad, (n_samples, n_dim), grad_target, [1])
        #gradient_error = tf.test.compute_gradient_error( ref_model.w, (1, n_kernels), grad_target, [1])
        #gradient_error = tf.test.compute_gradient_error( ref_model.mu, (1, n_kernels, n_dim), grad_target, [1])
        gradient_error = tf.test.compute_gradient_error( ref_model.sigma, (1, n_kernels, n_dim), grad_target, [1], delta=0.001)
    
        print("Gradient error: ", gradient_error)
    
    
    
    
    
    
    
    
    
    
    
    
    
