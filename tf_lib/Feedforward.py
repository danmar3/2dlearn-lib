##############################################################################
# 
# This file defines some common feedforward Neural Network arquitectures:
#      - ConvNet: convolutional neural network
#      - MlpNet: multilayer perceptron
#
# Created by: Daniel L. Marino (daniellml55@gmail.com)
#
# TODO: add saver for the networks
############################################################################


import numpy as np
import tensorflow as tf

class AlexnetLayer(object):
    '''Creates a layer like the one used in (ImageNet Classification 
    with Deep Convolutional Neural Networks).
    
    The format for filter_size is: [filter_size_dim0 , filter_size_dim1], it performs 2D convolution
    The format for n_maps is:      [num_input_maps, num_output_maps]
    
    The format for pool_size is:   [pool_size_dim0, pool_size_dim1]
    '''
    def __init__(self, filter_size, n_maps, pool_size, name=''):
        self.pool_size= pool_size
        self.weights = tf.Variable(
            tf.truncated_normal( [filter_size[0], filter_size[1], n_maps[0], n_maps[1]], stddev=0.1),
            name= 'W_'+name
        )
        
        self.bias = tf.Variable(tf.truncated_normal([n_maps[1]], stddev=0.1),
                                name= 'b_'+name
                               )
    def evaluate(self, input_tensor):
        # Perform Convolution
        # the layers performs a 2D convolution, with a strides of 1
        conv = tf.nn.conv2d(input_tensor, self.weights, strides=[1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv + self.bias)
        
        # Perform Pooling if the size of the pooling layer is bigger than 1
        # note that the size of the pooling kernel and the stride is the same
        if (self.pool_size[0]==1 and self.pool_size[1]==1):
            return hidden

        else:
            pool= tf.nn.max_pool(hidden, ksize=  [1, self.pool_size[0], self.pool_size[1], 1],
                                         strides=[1, self.pool_size[0], self.pool_size[1], 1], 
                                         padding='VALID')
            return pool
    
class FullyconnectedLayer(object):
    '''Standard fully connected layer'''
    def __init__(self, n_inputs, n_units, afunction= None, name=''):
        self.n_inputs= n_inputs
        self.n_units= n_units
        
        if afunction is None:
            self.afunction= tf.nn.relu
        else:
            self.afunction= afunction
        
        self.weights= tf.Variable(tf.truncated_normal([n_inputs, n_units], stddev=0.1), name= 'W_'+name)
        self.bias= tf.Variable(tf.truncated_normal([n_units], stddev=0.1), name= 'b_'+name)
        
        
    def evaluate(self, input_mat):
        return self.afunction(tf.matmul(input_mat, self.weights) + self.bias)        

class NetConf(object):
    '''This is a wrapper to any network configuration, it contains the references to
    the placeholders for inputs and labels, and the reference of the computation graph for 
    the network
    
    inputs: placeholder for the inputs
    labels: placeholder for the labels
    y: output of the comptuation graph (logits)
    loss: loss for the network
    '''
    def __init__(self, inputs, labels, y, loss):
        self.inputs= inputs
        self.labels= labels
        self.y= y
        self.loss= loss
        
class ConvNet(object):
    ''' Creates a Convolutional neural network 
    It creates a convolutional neural network similar to the one used 
    in (ImageNet Classification with Deep Convolutional Neural Networks)
    
    It performs a series of 2d Convolutions and pooling operations, then
    a standard fully connected stage and finaly a softmax
    
    input_size: [size_dim0, size_dim1]
    n_outputs: number of outputs
    n_input_maps: number of input maps
    n_filters: list with the number of filters for layer
    filter_size: list with the size of the kernel for each layer, 
                 the format for the size of each layer is: [filter_size_dim0 , filter_size_dim1] 
    pool_size: list with the size of the pooling kernel foreach layer,
               the format for each layer is: [pool_size_dim0, pool_size_dim1]
    n_hidden: list with the number of units on each fully connected layer
    
    '''
    def __init__(self, input_size, n_input_maps, n_outputs, 
                 n_filters, filter_size, 
                 pool_size, 
                 n_hidden, 
                 name=''):
        ''' All variables corresponding to the weights of the network are defined
        '''
        self.input_size= input_size
        self.n_input_maps= n_input_maps
        self.n_outputs= n_outputs
        self.n_filters= n_filters
        self.filter_size= filter_size
        self.pool_size= pool_size
        self.n_hidden= n_hidden
        
        # 1. Create the convolutional layers:
        self.conv_layers= list() 
        self.conv_layers.append( 
            AlexnetLayer(filter_size[0], [n_input_maps, n_filters[0]], pool_size[0], name= 'conv_0'+name) 
        )
        
        for l in range(1,len(n_filters)):
            self.conv_layers.append( 
                AlexnetLayer(filter_size[l], [n_filters[l-1], n_filters[l]], pool_size[l], name= 'conv_'+str(l)+name) 
            )
            
        # Get size after convolution phase
        final_size= [input_size[0], input_size[1]]
        for i in range(len(filter_size)):
            final_size[0]= (final_size[0] - 2*(filter_size[i][0]//2))//pool_size[i][0]
            final_size[1]= (final_size[1] - 2*(filter_size[i][1]//2))//pool_size[i][1]
        
        if final_size[0]==0:
            final_size[0]=1
        if final_size[1]==0:
            final_size[1]=1
        
        # 2. Create the fully connected layers:        
        self.full_layers= list()
        self.full_layers.append( 
            FullyconnectedLayer( final_size[0] * final_size[1] * n_filters[-1], n_hidden[0], name= 'full_0'+name) 
        )
        for l in range(1,len(n_hidden)):
            self.full_layers.append( 
                FullyconnectedLayer(n_hidden[l-1], n_hidden[l], name= 'full_'+str(l)+name) 
            )
        
        # 3. Create the final layer:
        self.linear_weights= tf.Variable(
            tf.truncated_normal( [n_hidden[-1], n_outputs], stddev=0.1),
            name= 'Wlin_'+name
        )
        
        self.linear_bias = tf.Variable(tf.truncated_normal([n_outputs], stddev=0.1),
                                name= 'blin_'+name
                               )

    def setup(self, batch_size, drop_prob= None, l2_reg_coef= None, loss_type= None):
        ''' Defines the computation graph of the neural network for a specific batch size 
        
        drop_prob: placeholder used for specify the probability for dropout. If this coefficient is set, then
                   dropout regularization is added between all fully connected layers(TODO: allow to choose which layers)
        l2_reg_coef: coeficient for l2 regularization
        loss_type: type of the loss being used for training the network, the options are:
                - 'cross_entropy': for classification tasks
                - 'l2': for regression tasks
        '''
        inputs= tf.placeholder( tf.float32, 
                                shape=(batch_size, self.input_size[0], self.input_size[1], self.n_input_maps ))
        
        labels= tf.placeholder( tf.float32, shape=(batch_size, self.n_outputs))
        
        
        # 1. convolution stage
        out= inputs
        for layer in self.conv_layers:
            out= layer.evaluate(out)
                
        # 2. fully connected stage
        # reshape
        shape = out.get_shape().as_list()
        print('Shape of input matrix entering to Fully connected layers:', shape)
        out = tf.reshape(out, [shape[0], shape[1] * shape[2] * shape[3]])
        # mlp
        for layer in self.full_layers:
            out = layer.evaluate(out)
            if drop_prob is not None:
                out = tf.nn.dropout(out, drop_prob)
        
        # 3. linear stage
        y = tf.matmul(out, self.linear_weights) + self.linear_bias
        
        # 4. loss # TODO: add number of parameters to loss so hyperparameters are more easy to tune, also put None as default and do not calculate loss if it is None
        # l2 regularizer 
        l2_reg= 0
        if l2_reg_coef is not None:
            for layer in self.full_layers:
                l2_reg += tf.nn.l2_loss(layer.weights)
            l2_reg = l2_reg_coef*l2_reg
            
        # loss
        if loss_type is None:
            loss= None
        elif loss_type=='cross_entropy':
            if self.n_outputs==1:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, labels)) + l2_reg
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, labels)) + l2_reg
        elif loss_type=='l2':
            loss = tf.reduce_mean(tf.nn.l2_loss(y-labels)) + l2_reg
            
        return NetConf(inputs, labels, y, loss)
        
class MlpNet(object):
    def __init__(self, n_inputs, n_outputs, n_hidden, afunction= None, name=''):
        '''All variables corresponding to the weights of the network are defined
        n_inputs: number of inputs
        n_outputs: number of outputs
        n_hidden: list with the number of hidden units in each layer
        full_layers: list of fully connected layers
        afunction: function, or list of functions specifying the activation function being used. if
                   not specified, the default is relu
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
            self.afunction= [tf.nn.relu for i in range(len(n_hidden))]
        
        # Create the fully connected layers:        
        self.full_layers= list()
        self.full_layers.append( 
            FullyconnectedLayer( n_inputs, n_hidden[0], 
                                 afunction= self.afunction[0], 
                                 name= 'full_0'+name) 
        )
        for l in range(1,len(n_hidden)):
            self.full_layers.append( 
                FullyconnectedLayer( n_hidden[l-1], n_hidden[l], 
                                     afunction= self.afunction[l], 
                                     name= 'full_'+str(l)+name) 
            )
        
        # Create the final layer:
        self.linear_weights= tf.Variable(
            tf.truncated_normal( [n_hidden[-1], n_outputs], stddev=0.1),
            name= 'Wlin_'+name
        )
        
        self.linear_bias = tf.Variable(tf.truncated_normal([n_outputs], stddev=0.1),
                                name= 'blin_'+name
                               )
        
    def setup(self, batch_size, drop_prob= None, l2_reg_coef= None, loss_type= None):
        ''' Defines the computation graph of the neural network for a specific batch size 
        
        drop_prob: placeholder used for specify the probability for dropout. If this coefficient is set, then
                   dropout regularization is added between all fully connected layers(TODO: allow to choose which layers)
        l2_reg_coef: coeficient for l2 regularization
        loss_type: type of the loss being used for training the network, the options are:
                - 'cross_entropy': for classification tasks
                - 'l2': for regression tasks
        '''
        inputs= tf.placeholder( tf.float32, 
                                shape=(batch_size, self.n_inputs ))
        
        labels= tf.placeholder( tf.float32, shape=(batch_size, self.n_outputs))
                
        # 1. fully connected stage
        out= inputs
        for layer in self.full_layers:
            out = layer.evaluate(out)
            if drop_prob is not None:
                out = tf.nn.dropout(out, drop_prob)
        
        # 2. linear stage
        y = tf.matmul(out, self.linear_weights) + self.linear_bias
        
        # 3. loss
        # l2 regularizer
        l2_reg= 0
        if l2_reg_coef is not None:
            for layer in self.full_layers:
                l2_reg += tf.nn.l2_loss(layer.weights)
            l2_reg = l2_reg_coef*l2_reg
            
        # loss
        if loss_type is None:
            loss= None
        elif loss_type=='cross_entropy':
            if self.n_outputs==1:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, labels)) + l2_reg
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, labels)) + l2_reg
        elif loss_type=='l2':
            loss = tf.reduce_mean(tf.nn.l2_loss(y-labels)) + l2_reg
            
        return NetConf(inputs, labels, y, loss)
        
        
        
        