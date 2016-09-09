##############################################################################
# 
# This file defines some common feedforward Neural Network arquitectures:
#      - ConvNet: convolutional neural network
#      - MlpNet: multilayer perceptron
#
# Created by: Daniel L. Marino (daniellml55@gmail.com)
#
############################################################################


import numpy as np
import tensorflow as tf


''' -------------------------------------------- Layers --------------------------------------------- '''

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
            name= 'w'+name
        )
        
        self.bias = tf.Variable(tf.truncated_normal([n_maps[1]], stddev=0.1),
                                name= 'b'+name
                               )
        
        # define the saver dictionary with the training parameters
        self.saver_dict= dict()
        self.saver_dict['w'+name] =  self.weights
        self.saver_dict['b'+name] =  self.bias
        
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
        

        
        
        
        

        
class StridedConvLayer(object):
    '''Creates a convolutional layer that uses strided convolutions instead of pooling.
    
    The format for filter_size is: [filter_size_dim0 , filter_size_dim1], it performs 2D convolution
    The format for n_maps is:      [num_input_maps, num_output_maps]
    
    The format for stride is:   [stride_dim0, stride_dim1]
    '''
    def __init__(self, filter_size, n_maps, stride, name=''):
        self.stride= stride
        self.weights = tf.Variable(
            tf.truncated_normal( [filter_size[0], filter_size[1], n_maps[0], n_maps[1]], stddev=0.1),
            name= 'w'+name
        )
        
        self.bias = tf.Variable(tf.truncated_normal([n_maps[1]], stddev=0.1),
                                name= 'b'+name
                               )
        
        # define the saver dictionary with the training parameters
        self.saver_dict= dict()
        self.saver_dict['w'+name] =  self.weights
        self.saver_dict['b'+name] =  self.bias
        
    def evaluate(self, input_tensor, padding= 'VALID'):
        # Perform Convolution
        # the layers performs a 2D convolution, with a strides of 1
        conv = tf.nn.conv2d(input_tensor, self.weights, strides=[1, self.stride[0], self.stride[1], 1], padding= padding)
        hidden = tf.nn.relu(conv + self.bias)
        
        return hidden

    

    
class ConvTransposeLayer(object):
    '''Creates a "deconvolutional" layer.
    
    The format for filter_size is: [filter_size_dim0 , filter_size_dim1], it performs 2D convolution
    The format for n_maps is:      [num_input_maps, num_output_maps]
    
    The format for stride is:   [stride_dim0, stride_dim1]
    '''
    def __init__(self, filter_size, n_maps, stride, name=''):
        self.stride= stride
        self.n_in_maps= n_maps[0]
        self.n_out_maps= n_maps[1]
        
        self.weights = tf.Variable(
            tf.truncated_normal( [filter_size[0], filter_size[1], n_maps[1], n_maps[0]], stddev=0.1),
            name= 'w'+name
        )
        
        self.bias = tf.Variable(tf.truncated_normal([n_maps[1]], stddev=0.1),
                                name= 'b'+name
                               )
        
        # define the saver dictionary with the training parameters
        self.saver_dict= dict()
        self.saver_dict['w'+name] =  self.weights
        self.saver_dict['b'+name] =  self.bias
        
    def evaluate(self, input_tensor, padding= 'SAME'):
        # Perform Convolution transpose
        # the layers performs a 2D convolution, with a strides of 1
        
        #conv = tf.nn.conv2d(input_tensor, self.weights, strides=[1, self.stride[0], self.stride[1], 1], padding= padding)
        
        in_shape= input_tensor.get_shape().as_list()
        out_shape= [in_shape[0], 
                    in_shape[1]*self.stride[0],
                    in_shape[2]*self.stride[1],
                    self.n_out_maps
                   ]
        deconv = tf.nn.conv2d_transpose(input_tensor, 
                                        self.weights, 
                                        output_shape= out_shape, 
                                        strides=[1, self.stride[0], self.stride[1], 1],
                                        padding= padding
                                       )
        
        hidden = tf.nn.relu(deconv + self.bias)
        
        return hidden
    
    
    
    
    

    
class FullyconnectedLayer(object):
    '''Standard fully connected layer'''
    def __init__(self, n_inputs, n_units, afunction= None, name=''):
        self.n_inputs= n_inputs
        self.n_units= n_units
        
        if afunction is None:
            self.afunction= tf.nn.relu
        else:
            self.afunction= afunction
        
        self.weights= tf.Variable(tf.truncated_normal([n_inputs, n_units], stddev=0.1), name= 'W'+name)
        self.bias= tf.Variable(tf.truncated_normal([n_units], stddev=0.1), name= 'b'+name)
        
        # define the saver dictionary with the training parameters
        self.saver_dict= dict()
        self.saver_dict['w'+name] =  self.weights
        self.saver_dict['b'+name] =  self.bias
        
    def evaluate(self, input_mat):
        return self.afunction(tf.matmul(input_mat, self.weights) + self.bias)        

 
    
    
    
    


class AffineLayer(object):
    '''Standard affine (W*X+b) fully connected layer'''
    def __init__(self, n_inputs, n_units, name=''):
        self.n_inputs= n_inputs
        self.n_units= n_units
        
        self.weights= tf.Variable(tf.truncated_normal([n_inputs, n_units], stddev=0.1), name= 'W'+name)
        self.bias= tf.Variable(tf.truncated_normal([n_units], stddev=0.1), name= 'b'+name)
        
        # define the saver dictionary with the training parameters
        self.saver_dict= dict()
        self.saver_dict['w'+name] =  self.weights
        self.saver_dict['b'+name] =  self.bias
                
    def evaluate(self, input_mat):
        return tf.matmul(input_mat, self.weights) + self.bias
    

    
    
    
class LinearLayer(object):
    '''Standard linear (W*X) fully connected layer'''
    def __init__(self, n_inputs, n_units, name=''):
        self.n_inputs= n_inputs
        self.n_units= n_units
        
        self.weights= tf.Variable(tf.truncated_normal([n_inputs, n_units], stddev=0.1), name= 'W'+name)
                
        # define the saver dictionary with the training parameters
        self.saver_dict= dict()
        self.saver_dict['w'+name] =  self.weights
                        
    def evaluate(self, input_mat):
        return tf.matmul(input_mat, self.weights) 
    

    
    
    
''' -------------------------------------------- Networks --------------------------------------------- '''
   
    
class NetConf(object):
    '''This is a wrapper to any network configuration, it contains the references to
    the placeholders for inputs and labels, and the reference of the computation graph for 
    the network
    
    inputs: placeholder for the inputs
    labels: placeholder for the labels
    y: output of the comptuation graph, ussually a linear map from the last layer (logits)
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
    
    input_size: size of the input maps: [size_dim0, size_dim1]
    n_outputs: number of outputs
    n_input_maps: number of input maps
    n_filters: list with the number of filters for layer
    filter_size: list with the size of the kernel for each layer, 
                 the format for the size of each layer is: [filter_size_dim0 , filter_size_dim1] 
    pool_size: list with the size of the pooling kernel foreach layer,
               the format for each layer is: [pool_size_dim0, pool_size_dim1]
    n_hidden: list with the number of units on each fully connected layer
    
    out_conv_shape: size of the output map from the convolution stage: [size_dim0, size_dim1]
    
    '''
    def __init__(self, input_size, n_input_maps, n_outputs, 
                 n_filters, filter_size, 
                 pool_size, 
                 n_hidden= [], 
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
            AlexnetLayer(filter_size[0], [n_input_maps, n_filters[0]], pool_size[0], name= '0_conv_'+name) 
        )
        
        for l in range(1,len(n_filters)):
            self.conv_layers.append( 
                AlexnetLayer(filter_size[l], [n_filters[l-1], n_filters[l]], pool_size[l], name= str(l)+'_conv_'+name) 
            )
        
        
        
        # Get size after convolution phase
        final_size= [input_size[0], input_size[1]]
        for i in range(len(filter_size)):
            final_size[0]= (final_size[0] - (filter_size[i][0]-1))//pool_size[i][0]
            final_size[1]= (final_size[1] - (filter_size[i][1]-1))//pool_size[i][1]
        
        if final_size[0]==0:
            final_size[0]=1
        if final_size[1]==0:
            final_size[1]=1
        self.out_conv_shape = final_size
        print("Shape of the maps after convolution stage:", self.out_conv_shape)
        
        # 2. Create the fully connected layers:        
        if len(n_hidden)>0:
            self.full_layers= list()
            self.full_layers.append( 
                FullyconnectedLayer( final_size[0] * final_size[1] * n_filters[-1], n_hidden[0], name= '0_full_'+name) 
            )
            for l in range(1,len(n_hidden)):
                self.full_layers.append( 
                    FullyconnectedLayer(n_hidden[l-1], n_hidden[l], name= str(l)+'_full_'+name) 
                )

            # 3. Create the final layer:
            self.out_layer =  AffineLayer( n_hidden[-1], n_outputs, name= '_lin_'+name)
        
        elif (n_outputs is not None):
            # 3. Create the final layer: (TODO: test this!!!!!!)
            self.out_layer =  AffineLayer( final_size[0] * final_size[1] * n_filters[-1], n_outputs, name= '_lin_'+name)
            
        
        # 4. Define the saver for the weights of the network
        saver_dict= dict()
        for l in range(len(self.conv_layers)):
            saver_dict.update( self.conv_layers[l].saver_dict )          
        
        if len(n_hidden)!=0:
            for l in range(len(self.full_layers)):
                saver_dict.update( self.full_layers[l].saver_dict )               
        elif (n_outputs is not None):        
            saver_dict.update( self.out_layer.saver_dict )            
        
        self.saver= tf.train.Saver(saver_dict)
        
        
        
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
        
        if (loss_type is not None) or (len(self.n_hidden)==0):
            labels= tf.placeholder( tf.float32, shape=(batch_size, self.n_outputs))
        else:
            labels= None
        
        
        # 1. convolution stage
        out= inputs
        for layer in self.conv_layers:
            out= layer.evaluate(out)
                
        # 2. fully connected stage
        # 2.1 reshape
        shape = out.get_shape().as_list()
        print('Shape of input matrix entering to Fully connected layers:', shape)
        out = tf.reshape(out, [shape[0], shape[1] * shape[2] * shape[3]])
        
        # if no fully connected layers, return here:
        if (len(self.n_hidden)==0 and  self.n_outputs is None):
            return NetConf(inputs, None, out, None)
        elif len(self.n_hidden)==0: #TODO: check and add loss in this case
            out = self.out_layer.evaluate(out)
            return NetConf(inputs, None, out, None)
                
        # 2.2 mlp
        for layer in self.full_layers:
            out = layer.evaluate(out)
            if drop_prob is not None:
                out = tf.nn.dropout(out, drop_prob)
        
        # 3. linear stage
        y = self.out_layer.evaluate(out)
        
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
            self.afunction= [tf.nn.relu for i in range(len(n_hidden))]
        
        # Create the fully connected layers:        
        self.full_layers= list()
        self.full_layers.append( 
            FullyconnectedLayer( n_inputs, n_hidden[0], 
                                 afunction= self.afunction[0], 
                                 name= '0_full_'+name) 
        )
        for l in range(1,len(n_hidden)):
            self.full_layers.append( 
                FullyconnectedLayer( n_hidden[l-1], n_hidden[l], 
                                     afunction= self.afunction[l], 
                                     name= str(l)+'_full_'+name) 
            )
        
        # Create the final layer:
        self.out_layer= AffineLayer( n_hidden[-1], n_outputs, name= '_lin_'+name)
        
                
        # 4. Define the saver for the weights of the network
        saver_dict= dict()            
        for l in range(len(self.full_layers)):
            saver_dict.update( self.full_layers[l].saver_dict )
                        
        saver_dict.update( self.out_layer.saver_dict )
                        
        self.saver= tf.train.Saver(saver_dict)
        
        
    def setup(self, batch_size, drop_prob= None, l2_reg_coef= None, loss_type= None, inputs= None):
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
        
                        
        # 1. fully connected stage
        out= inputs
        for layer in self.full_layers:
            out = layer.evaluate(out)
            if drop_prob is not None:
                out = tf.nn.dropout(out, drop_prob)
        
        # 2. linear stage
        if drop_prob is not None:
            y = self.out_layer.evaluate(tf.nn.dropout(out, drop_prob))
        else:
            y = self.out_layer.evaluate(out)
        
        # 3. loss
        # l2 regularizer
        l2_reg= 0
        if l2_reg_coef is not None:
            for layer in self.full_layers:
                l2_reg += tf.nn.l2_loss(layer.weights) 
            l2_reg += tf.nn.l2_loss(self.out_layer.weights)
            l2_reg = l2_reg_coef*l2_reg
            
        # loss
        if loss_type is None:
            loss= None
            labels= None
        elif loss_type=='cross_entropy':
            labels= tf.placeholder( tf.float32, shape=(batch_size, self.n_outputs))
            
            if self.n_outputs==1:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, labels)) + l2_reg
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, labels)) + l2_reg
        elif loss_type=='l2':
            labels= tf.placeholder( tf.float32, shape=(batch_size, self.n_outputs))
            
            loss = tf.reduce_mean(tf.nn.l2_loss(y-labels)) + l2_reg
            
        return NetConf(inputs, labels, y, loss)
        
        
        
        
        
        
        
        
        
class StridedConvNet(object):   # TODO!!!!!!!!!!!!!!!!!!!
    ''' Creates a Convolutional neural network using strided convolutions. It does not use pooling
    
    It performs a series of 2d strided Convolution operations, then
    a standard fully connected stage and finaly an affine mapping
    
    input_size: size of the input maps: [size_dim0, size_dim1]
    n_outputs: number of outputs
    n_input_maps: number of input maps
    n_filters: list with the number of filters for layer
    filter_size: list with the size of the kernel for each layer, 
                 the format for the size of each layer is: [filter_size_dim0 , filter_size_dim1] 
    strides: list with the size of the strides for each layer,
               the format for each layer is: [stride_dim0, stride_dim1]
    n_hidden: list with the number of units on each fully connected layer
    
    out_conv_shape: size of the output map from the convolution stage: [size_dim0, size_dim1]
    
    '''
    def __init__(self, input_size, n_input_maps, n_outputs, 
                 n_filters, filter_size, 
                 strides, 
                 n_hidden= [], 
                 name=''):
        ''' All variables corresponding to the weights of the network are defined
        '''
        self.input_size= input_size
        self.n_input_maps= n_input_maps
        self.n_outputs= n_outputs
        self.n_filters= n_filters
        self.filter_size= filter_size
        self.strides= strides
        self.n_hidden= n_hidden
        
        # 1. Create the convolutional layers:
        self.conv_layers= list() 
        self.conv_layers.append( 
            StridedConvLayer(filter_size[0], [n_input_maps, n_filters[0]], strides[0], name= '0_conv_'+name) 
        )
        
        for l in range(1,len(n_filters)):
            self.conv_layers.append( 
                StridedConvLayer(filter_size[l], [n_filters[l-1], n_filters[l]], strides[l], name= str(l)+'_conv_'+name) 
            )
        
        
        
        # Get size after convolution phase
        final_size= [input_size[0], input_size[1]]
        for i in range(len(filter_size)):
            final_size[0]= (final_size[0] - (filter_size[i][0]-1))//strides[i][0]
            final_size[1]= (final_size[1] - (filter_size[i][1]-1))//strides[i][1]
        
        if final_size[0]==0:
            final_size[0]=1
        if final_size[1]==0:
            final_size[1]=1
        self.out_conv_shape = final_size
        print("Shape of the maps after convolution stage:", self.out_conv_shape)
        
        # 2. Create the fully connected layers: 
        self.full_layers= list()
        if len(n_hidden)>0:
            self.full_layers.append( 
                FullyconnectedLayer( final_size[0] * final_size[1] * n_filters[-1], n_hidden[0], name= '0_full_'+name) 
            )
            for l in range(1,len(n_hidden)):
                self.full_layers.append( 
                    FullyconnectedLayer(n_hidden[l-1], n_hidden[l], name= str(l)+'_full_'+name) 
                )

            # 3. Create the final layer:
            self.out_layer =  AffineLayer( n_hidden[-1], n_outputs, name= '_lin_'+name)
            
        elif (n_outputs is not None):
            # 3. Create the final layer: (TODO: test this!!!!!!)
            self.out_layer =  AffineLayer( final_size[0] * final_size[1] * n_filters[-1], n_outputs, name= '_lin_'+name)
            
            
            
        # 4. Define the saver for the weights of the network
        saver_dict= dict()
        for l in range(len(self.conv_layers)):
            saver_dict.update( self.conv_layers[l].saver_dict )          
        
        if len(n_hidden)!=0:
            for l in range(len(self.full_layers)):
                saver_dict.update( self.full_layers[l].saver_dict )               
        elif (n_outputs is not None):   
            saver_dict.update( self.out_layer.saver_dict )            
        
        self.saver= tf.train.Saver(saver_dict)
        
        
        
    def setup(self, batch_size, drop_prob= None, l2_reg_coef= None, loss_type= None, inputs= None):
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
                                    shape=(batch_size, self.input_size[0], self.input_size[1], self.n_input_maps ))
        
        if (loss_type is not None) or (len(self.n_hidden)==0):
            labels= tf.placeholder( tf.float32, shape=(batch_size, self.n_outputs))
        else:
            labels= None
        
        
        # 1. convolution stage
        out= inputs
        for layer in self.conv_layers:
            out= layer.evaluate(out)
                
        # 2. fully connected stage
        # 2.1 reshape
        shape = out.get_shape().as_list()
        print('Shape of input matrix entering to Fully connected layers:', shape)
        out = tf.reshape(out, [shape[0], shape[1] * shape[2] * shape[3]])
        
        # if no fully connected layers, return here:
        if (len(self.n_hidden)==0 and self.n_outputs is None):
            return NetConf(inputs, None, out, None)
        
                
        # 2.2 mlp
        for layer in self.full_layers:
            out = layer.evaluate(out)
            if drop_prob is not None:
                out = tf.nn.dropout(out, drop_prob)
        
        # 3. linear stage
        y = self.out_layer.evaluate(out)
        
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
    
    
    
    
    
    
    
class StridedDeconvNet(object):   # TODO!!!!!!!!!!!!!!!!!!!
    ''' Creates a Deconvolutional neural network using upsampling
    
    It performs a "deconvolutional" neural network similar to the one used
    in "UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS"
    (http://arxiv.org/pdf/1511.06434v2.pdf)
    
    The network maps a vector of size n_inputs to a 2d map with several chanels
    
    First a linear mapping is performed, then a reshape to form an initial tensor of 2d maps with chanels,
    then a series of upscaling and convolutions are performed
    
    n_inputs: size of the input vectors
    input_size: size of the maps after linear stage: [size_dim0, size_dim1]
    n_input_maps: number of maps after linear stage
    n_filters: list with the number of filters for each layer
    filter_size: list with the size of the kernel for each layer, 
                 the format for the size of each layer is: [filter_size_dim0 , filter_size_dim1] 
    upsampling: list with the size for the upsampling in each deconv layer: 
                [upsampling_dim0, upsampling_dim1]
        
    
    
    in_layer: input layer, a linear layer for mapping the inputs to the desired output
    
    '''
    def __init__(self, n_inputs, input_size, n_input_maps,
                 n_filters, filter_size, 
                 upsampling, 
                 name=''):
        ''' All variables corresponding to the weights of the network are defined
        '''
        self.n_inputs= n_inputs
        self.input_size= input_size 
        self.n_input_maps= n_input_maps        
        self.n_filters= n_filters
        self.filter_size= filter_size
        self.upsampling= upsampling
           
        # 1. Create the linear layer 
        self.in_layer= LinearLayer( n_inputs, n_input_maps*input_size[0]*input_size[1], 
                                    name= '_lin_'+name)
        
        # 2. Create the convolutional layers:
        self.conv_layers= list()
        self.conv_layers.append( 
            ConvTransposeLayer(filter_size[0], [n_input_maps, n_filters[0]], upsampling[0], name= '0_conv_'+name) 
        )
        for l in range(1,len(n_filters)):
            self.conv_layers.append( 
                ConvTransposeLayer(filter_size[l], [n_filters[l-1], n_filters[l]], upsampling[l], name= str(l)+'_conv_'+name) 
            )
        
        # 4. Define the saver for the weights of the network
        saver_dict= dict()
        for l in range(len(self.conv_layers)):
            saver_dict.update( self.conv_layers[l].saver_dict )                     
        
        self.saver= tf.train.Saver(saver_dict)
        
        
        
    def setup(self, batch_size, drop_prob= None):
        ''' Defines the computation graph of the neural network for a specific batch size 
        
        drop_prob: placeholder used for specify the probability for dropout. If this coefficient is set, then
                   dropout regularization is added between all fully connected layers(TODO: allow to choose which layers)
        '''
        inputs= tf.placeholder( tf.float32, 
                                shape=(batch_size, self.n_inputs) )
        
        
        # 1. linear stage
        out = self.in_layer.evaluate(inputs)
                
        # 1.1 reshape
        shape = out.get_shape().as_list()
        out = tf.reshape(out, [shape[0], 
                               self.input_size[0], 
                               self.input_size[1], 
                               self.n_input_maps]
                        )
               
        # 2. convolution stage
        for layer in self.conv_layers:
            
            out= layer.evaluate(out, 'SAME')
                        
        return NetConf(inputs, None, out, None)
                
        
        