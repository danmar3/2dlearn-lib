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
#   Description: This file defines a gaussian mixture layer, 
#
#   ***********************************************************************

import numpy as np
import tensorflow as tf
import math
from twodlearn.tf_lib.Feedforward import *
import twodlearn.tf_lib.ops as tdl


global global_test

class GmmParamsLayer(object):
    '''This layer transforms a vector into a valid set of 
    parameters that can be fed to a GMM model.   
    '''
    def __init__(self, n_dim, n_kernels= 1, diagonal= True, name=''):
        
        self.n_dim = n_dim
        self.n_kernels = n_kernels
        self.diagonal = diagonal
                
        
        
    def evaluate(self, x):
        # x: input tensor that represents the parameters of the gmm. 
        #    format: [sample_id, parameters_for_current_sample]
        #    parameters_for_current_sample: [mu sigma w]
        
        # TODO: assert (x.get_shape())
        
        # Get parameters into the input format of GmmLayer
        if self.diagonal == True:
            mu = tf.slice(x, [0, 0], [-1, self.n_kernels*self.n_dim], name=None)
            mu = tf.reshape(mu, [-1, self.n_kernels, self.n_dim])
            
            sigma = tf.slice(x, [0, self.n_kernels*self.n_dim], [-1, (self.n_kernels*self.n_dim)], name=None)
            sigma = tf.reshape(sigma, [-1, self.n_kernels, self.n_dim])
            sigma_t = tf.exp(sigma)
            
            w = tf.slice(x, [0, 2*self.n_kernels*self.n_dim], [-1, -1], name=None)
            w_t= tf.nn.softmax(w)
                        
        # TODO: implement non-diagonal gmm
        
        return mu, sigma_t, w_t
              
        
class GmmModel(object):
    ''' Evaluates the gaussian mixture probability density function
    given the parameters of the model mu, sigma, and w. 
    sigma must be positive definite (at the moment only diagonal)
    w must be proper, i.e. sum to one and its elements must be positive
    '''
    def __init__(self,  n_dim, n_kernels= 1, diagonal= True, method='tf' , name=''):
        self.n_dim = n_dim
        self.n_kernels = n_kernels
        self.diagonal = diagonal
        self.method = method
        
    def evaluate(self, y, mu, sigma, w):
        #     - y: [sample_id, dim]
        #     - mu: 3d tensor containing the means for the gaussians.
        #           the "depth" dimention (3rd) is used to index the
        #           gaussians.    [sample_id, kernel_id, dim]
        #     - sigma: 3d tensor containing the covariance matrix of the
        #              gaussians. [sample_id, kernel_id, dim] for diagonal matrices
        #     - w: vector in form of a 3d tensor containing the weights
        #          for each one of the gaussians, they have to sum one. 
        #          [sample_id, kernel_id]
        
        if (self.diagonal == True) and (self.method =='tf'):
            norm_const = tf.inv( tf.sqrt((math.pow(2*np.pi, self.n_dim)) * tf.reduce_prod(sigma, 2))) # shape: [sample_id, kernel_id]
            
            sigma_inv = tf.inv( sigma ) # 1/x element-wise, shape: [sample_id, kernel_id, sigma...]
            
            x_mu = tf.reshape(y, [-1, 1, self.n_dim]) - mu # shape: [sample_id, kernel_id, x-mu]
            
            sigma_inv_x_mu = tf.mul( x_mu, sigma_inv )
            
            gaussians = tf.mul( norm_const, tf.exp( -0.5* tf.reduce_sum( x_mu * sigma_inv_x_mu, 2 ) ) ) #[sample_id, kernel_id]
                        
            y = tf.reduce_sum( tf.mul( w, gaussians  ), 1 )

        elif (self.diagonal == True) and (self.method =='tdl'):
            y,_,_ = tdl.gmm_model(y, w, mu, sigma)
        
        
        # TODO: non-diagonal covariances
        
        return y
        

class GmmNetConf(object):
    '''This is a wrapper to any network configuration, it contains the references to
    the placeholders for inputs and labels, and the reference of the computation graph for 
    the network
    
    inputs: placeholder for the inputs
    labels: placeholder for the labels
    y: output of the comptuation graph (logits)
    loss: loss for the network
    '''
    def __init__(self, inputs, labels, out, loss, mu, sigma, w):
        self.inputs= inputs
        self.labels= labels
        self.out= out
        self.loss= loss
        self.mu = mu
        self.sigma = sigma
        self.w = w

class GmmLayer(object):
    ''' Defines a GMM layer, which consists of a GmmParamsLayer and a GmmModelLayer
    GmmParamsLayer takes unconstrained parameters and transform them into a set of 
    valid Gmm parameters
    
    GmmModelLayer computes the value of the pdf for a given dataset based on the valid
    set of parameters
    '''
    def __init__(self, n_dim, n_kernels= 1, diagonal= True, method= 'tf', name=''):
        self.n_dim = n_dim
        self.n_kernels = n_kernels
        self.diagonal = diagonal
        
        self.gmm_params= GmmParamsLayer( n_dim, n_kernels, diagonal)
        self.gmm_model= GmmModel( n_dim, n_kernels, diagonal, method)
        
    def evaluate(self, y, unc_params):
        ''' y: dataset 
            unc_params: 2d matrix containing the set of unconstrained parameters 
        '''
        # transform parameters to 'valid' parameters
        mu_v, sigma_v, w_v = self.gmm_params.evaluate( unc_params )

        # evaluate the gaussian mixture model using the 'valid' parameters
        gmm_out = self.gmm_model.evaluate( y, mu_v, sigma_v, w_v )
        
        return gmm_out, mu_v, sigma_v, w_v


        
class GmmShallowModel(object):
    ''' Defines an standard GMM model, with constant mu, sigma and w parameters
    The GMM models a pdf of the form x ~ sum_i( w_i*N_i(x| mu_i, sigma_i) )
    '''
    def __init__(self, n_dim, n_kernels= 1, diagonal= True, method='tf', name=''):
        self.n_dim = n_dim
        self.n_kernels = n_kernels
        self.diagonal = diagonal
        
        self.unc_params = tf.Variable(tf.truncated_normal( [1, n_kernels + 2*n_kernels*n_dim ], stddev=0.001),
                                      name= 'gmm_unc_params'+name
                                     )
        
        self.gmm_layer= GmmLayer( n_dim, n_kernels, diagonal, method)
        
    def loss_eval(self, net_out):
        tol= 1e-9
        return tf.reduce_sum(-tf.log(net_out + tol))
        
        
    def setup(self, batch_size, x= None):
        if x is None:
            x= tf.placeholder( tf.float32, 
                               shape=(batch_size, self.n_dim ))
        
        out, mu_v, sigma_v, w_v = self.gmm_layer.evaluate( x, self.unc_params )
               
        return GmmNetConf(x, None, out, self.loss_eval(out), mu_v, sigma_v, w_v)
        
        
class GmmMlpModel(object):
    ''' Defines a GMM model for conditional pdf's, with mu, sigma and w parameters defined by a MLP network
    '''
    def __init__(self, n_inputs, n_outputs, n_hidden, n_kernels, afunction= None, diagonal=True, method='tf', name=''):
        self.n_inputs= n_inputs
        self.n_outputs= n_outputs
        self.n_hidden= n_hidden
        
        self.n_kernels = n_kernels
        self.diagonal = diagonal
        
        if diagonal==True:
            self.mlp_model = MlpNet( n_inputs, n_kernels + 2*n_kernels*n_outputs, 
                                     n_hidden, afunction, name= name+'_FF' )

            self.gmm_layer = GmmLayer(n_inputs, n_kernels, diagonal, method)
            
        # TODO: implement non-diagonal convariances
        
    def loss_eval(self, net_out):
        tol= 1e-9
        return tf.reduce_mean(-tf.log(net_out + tol))
    
    def setup(self, batch_size, inputs= None): 
        
        labels= tf.placeholder( tf.float32, 
                                shape=(batch_size, self.n_outputs ))
        
        mlp_conf = self.mlp_model.setup(batch_size, inputs= inputs)
        
        out, mu_v, sigma_v, w_v = self.gmm_layer.evaluate( labels, mlp_conf.y )
        
        return GmmNetConf(mlp_conf.inputs, labels, out, self.loss_eval(out), mu_v, sigma_v, w_v)
        
        
