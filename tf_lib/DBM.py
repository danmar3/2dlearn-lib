#************************************************************************
#      __   __  _    _  _____   _____
#     /  | /  || |  | ||     \ /  ___|
#    /   |/   || |__| ||    _||  |  _
#   / /|   /| ||  __  || |\ \ |  |_| |
#  /_/ |_ / |_||_|  |_||_| \_\|______|
#    
# 
#   Written by < Daniel L. Marino (marinodl@vcu.edu) > (2016)
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
#   Description: This file defines a deep restricted boltzman machine
#
#   ***********************************************************************

import numpy as np
import tensorflow as tf

def bernoulli_sample_tf(x):
    
    in_shape= x.get_shape().as_list()
    
    uniform_samp = np.random.rand(in_shape[0], in_shape[1])
    
    return tf.to_float(tf.greater(x, uniform_samp))


class RBM(object):
    '''Standard restricted boltzman machine'''
    def __init__(self, n_inputs, n_units, afunction= None, name=''):
        self.n_inputs= n_inputs
        self.n_units= n_units
        
        if afunction is None:
            self.afunction= tf.sigmoid
        else:
            self.afunction= afunction
        
        self.weights= tf.Variable(tf.truncated_normal([n_inputs, n_units], stddev=0.1), name= 'W'+name)
        self.bias_b= tf.Variable(tf.truncated_normal([n_units], stddev=0.1), name= 'b'+name)
        
        self.bias_c= tf.Variable(tf.truncated_normal([n_inputs], stddev=0.1), name= 'c'+name)
        
        # define the saver dictionary with the training parameters
        self.saver_dict= dict()
        self.saver_dict['w'+name] =  self.weights
        self.saver_dict['b'+name] =  self.bias_b
        self.saver_dict['c'+name] =  self.bias_c
        
        # define list of parameters
        self.params = [self.weights, self.bias_b, self.bias_c]
        
    def evaluate_h_given_x(self, input_mat):
        return self.afunction(tf.matmul(input_mat, self.weights) + self.bias_b)        
    
    
    def evaluate_x_given_h(self, input_mat):
        return self.afunction(tf.matmul(input_mat, tf.transpose(self.weights)) + self.bias_c)        
    
    
    def gibbs_sampling_given_h(self, h_prob, k=1):
        ''' generates a sample x after k gibbs samples
            h is assumed to be between [0-1], it is a set of probabilities
        '''
        h_samp = bernoulli_sample_tf(h_prob)
        x_prob = self.evaluate_x_given_h(h_samp)
               
        for i in range(k):
            x_samp = bernoulli_sample_tf(x_prob)
            h_prob = self.evaluate_h_given_x(x_samp)
            h_samp = bernoulli_sample_tf(h_prob)
            x_prob = self.evaluate_x_given_h(h_samp)
        
        return bernoulli_sample_tf(x_prob)
        
        
    def gibbs_sampling_given_x(self, x_prob, k=1):
        ''' generates a sample x after k gibbs samples
            x is assumed to be between [0-1], it is a set of probabilities 
        '''
        
        for i in range(k):
            x_samp = bernoulli_sample_tf(x_prob)
            h_prob = self.evaluate_h_given_x(x_samp)
            h_samp = bernoulli_sample_tf(h_prob)
            x_prob = self.evaluate_x_given_h(h_samp)
        
        return bernoulli_sample_tf(x_prob)
    
    def evaluate_cd_step(self, x, k=1, alpha= 0.001):
        ''' runs a step of the contrastive divergence algorithm using k gibbs samples
        '''
        in_shape= x.get_shape().as_list()
        
        x_samp = bernoulli_sample_tf(x)
        # generate a negative sample using gibbs sampling:
        xp_samp= self.gibbs_sampling_given_x( x, k)
        
        # parameter update
        h_x = self.evaluate_h_given_x(x_samp)
        h_xp = self.evaluate_h_given_x(xp_samp)
                
        dw = (1.0/in_shape[0])*( tf.matmul( tf.transpose(x_samp), h_x ) - 
                                 tf.matmul( tf.transpose(xp_samp), h_xp ))
        
        db = tf.reduce_mean( h_x - h_xp, reduction_indices= 0 )
        dc = tf.reduce_mean( x_samp - xp_samp, reduction_indices= 0 )
        
        op_W= self.weights.assign_add( alpha*dw )
        op_b= self.bias_b.assign_add( alpha*db )
        op_c= self.bias_c.assign_add( alpha*dc )
        
        # update parameters
        return tf.group( op_W, op_b, op_c ), xp_samp
        
class AutoencoderNetConf(object):
    '''This is a wrapper to any network configuration, it contains the references to
    the placeholders for inputs and labels, and the reference of the computation graph for 
    the network
    
    inputs: placeholder for the inputs
    labels: placeholder for the labels
    y: output of the comptuation graph (logits)
    loss: loss for the network
    
    xp_list: list with the "reconstructed" outputs using the transpose of the autoencoder layer
    h_list: list with the hidden layer outputs
    xp_losses: list with the loses for each one of the layers reconstructions
    '''
    def __init__(self, inputs, labels, y, loss, xp_list, h_list, xp_losses):
        self.inputs= inputs
        self.labels= labels
        self.y= y
        self.loss= loss
        
        self.xp_list = xp_list
        self.h_list = h_list
        self.xp_losses = xp_losses
        
    
class StackedAutoencoderNet(object):
    def __init__(self, n_inputs, n_outputs, n_hidden, afunction= None, name=''):
        '''All variables corresponding to the weights of the network are defined
        n_inputs: number of inputs
        n_outputs: number of outputs
        n_hidden: list with the number of hidden units in each layer
        full_layers: list of fully connected layers
        afunction: function, or list of functions specifying the activation function being used. if
                   not specified, the default is relu
        out_layer: output layer, for the moment, linear layer
        
        
        '''
        self.n_inputs= n_inputs
        self.n_hidden= n_hidden
        self.n_outputs= n_outputs
                
        # Check activation function being used
        if afunction is not None:
            if isinstance(afunction, list):
                self.afunction= afunction
            else:
                self.afunction= [afunction for i in range(len(n_hidden))]
        else:
            self.afunction= [tf.sigmoid for i in range(len(n_hidden))]
        
        # Create the fully connected layers:        
        self.full_layers= list()
        self.full_layers.append( 
            FullyconnectedLayer( n_inputs, n_hidden[0], 
                                 afunction= self.afunction[0], 
                                 name= '0_full_'+name) 
        )
        for l in range(1,len(n_hidden)):
            self.full_layers.append( 
                AutoencoderLayer( n_hidden[l-1], n_hidden[l], 
                                  afunction= self.afunction[l], 
                                  name= str(l)+'_full_'+name) 
            )
        
        # Create the final layer:
        self.out_layer= LinearLayer( n_hidden[-1], n_outputs, name= '_lin_'+name)
        
                
        # 4. Define the saver for the weights of the network
        saver_dict= dict()            
        for l in range(len(self.full_layers)):
            saver_dict.update( self.full_layers[l].saver_dict )
                        
        saver_dict.update( self.out_layer.saver_dict )
                        
        self.saver= tf.train.Saver(saver_dict)
    
    def compute_pred_loss(self, x, xp):
        return tf.reduce_mean(tf.nn.l2_loss(x-xp)) 
        
        
    def compute_output_loss(self, y, labels):
        # for cross-entropy loss
        #if self.n_outputs==1:
        #    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, labels)) + l2_reg
        #else:
        #    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, labels)) + l2_reg
        
        # for l2 loss:    
        return tf.reduce_mean(tf.nn.l2_loss(y-labels)) 
        
    
    def add_noise(self, input_tensor, l_idx):
        return input_tensor
        
        
    def setup(self, batch_size, drop_prob= None, l2_reg_coef= None, inputs= None):
        ''' Defines the computation graph of the neural network for a specific batch size 
        
        drop_prob: placeholder used for specify the probability for dropout. If this coefficient is set, then
                   dropout regularization is added between all fully connected layers(TODO: allow to choose which layers)
        l2_reg_coef: coeficient for l2 regularization
        loss_type: type of the loss being used for training the network, the options are:
                - 'cross_entropy': for classification tasks
                - 'l2': for regression tasks
        '''
        if inputs is None:
            inputs= tf.placeholder( tf.float32, 
                                    shape=(batch_size, self.n_inputs ))
        
        labels= tf.placeholder( tf.float32, shape=(batch_size, self.n_outputs))
                
        # 1. forward computation stage
        h_list = list()
        xp_list = list()
        
        h_list.append( self.full_layers[0].evaluate( self.add_noise( inputs, 0 ) ) )
        xp_list.append( self.full_layers[0].evaluate_transpose(h_list[-1]) )
                
        for l_idx in range(1,len(self.full_layers)):
                      
            if drop_prob is not None:
                h_list.append( 
                    self.full_layers[l_idx].evaluate( 
                                        tf.nn.dropout(
                                            self.add_noise( h_list[-1], l_idx ) , 
                                            drop_prob))
                             )
            else:
                h_list.append( self.full_layers[l_idx].evaluate( self.add_noise( h_list[-1], l_idx ) ) )
                
            xp_list.append( self.full_layers[l_idx].evaluate_transpose(h_list[-1]) )
            
        
        
        # 2. linear stage
        if drop_prob is not None:
            y = self.out_layer.evaluate(tf.nn.dropout(h_list[-1], drop_prob))
        else:
            y = self.out_layer.evaluate(h_list[-1])
        
        # 3. loss
        # l2 regularizer
        l2_reg= 0
        if l2_reg_coef is not None:
            for layer in self.full_layers:
                l2_reg += tf.nn.l2_loss(layer.weights) 
            l2_reg += tf.nn.l2_loss(self.out_layer.weights)
            l2_reg = l2_reg_coef*l2_reg
            
        # reconstruction loss
        xp_losses= list()
        xp_losses.append(self.compute_pred_loss(inputs, xp_list[0]))
        
        for i_xp in range(1,len(xp_list)):
            xp_losses.append(self.compute_pred_loss(h_list[i_xp-1], xp_list[i_xp]))
        
        # output loss
        loss= self.compute_output_loss(y, labels) + l2_reg
        
                
        return AutoencoderNetConf(inputs, labels, y, loss, xp_list, h_list, xp_losses)
    
    
    def get_pred_optimizers(self, NetConf, learning_rate=0.0002, beta1=0.5):
        opt_list = list()
        
        for l in range(len(NetConf.xp_losses)):
            xp_l= NetConf.xp_losses[l]
            
            opt_list.append( 
                tf.train.AdamOptimizer(
                    learning_rate= learning_rate, beta1= beta1
                ).minimize(xp_l, var_list = self.full_layers[l].params) 
            )
        
        return opt_list
        
        
        