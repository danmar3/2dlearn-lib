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
#   Description: This file defines some common LSTM network architectures
#
#   ***********************************************************************


import numpy as np
import tensorflow as tf
from twodlearn.tf_lib.Feedforward import AffineLayer 

# TODO: change the saver to be defined layer by layer

class LstmLayer:
    def __init__( self, FinalOut, inputs_list, 
                  last_output, last_state, labels_list, 
                  saved_output, saved_state, 
                  assign_saved_out_state, reset_saved_out_state,
                  error_per_sample): 
        self.y= FinalOut      # in the case of classification, this are the logits, i.e. the output from the final linear transformation
        self.inputs_list= inputs_list # list of tensor placeholders for the inputs
        self.labels_list= labels_list # list of tensor placeholders for the labels
        
        self.last_output= last_output # list of tensor variables for the last output of the unrolling
        self.last_state= last_state   # list of tensor variables for the last state of the unrolling
    
        self.saved_output= saved_output  # list of tensor placeholders for the inputs
        self.saved_state= saved_state    # list of tensor placeholders for the inputs
        
        self.assign_saved_out_state= assign_saved_out_state   # assigns last state and output to saved_state and saved_output
        
        ''''''
        self.reset_saved_out_state = tf.group( *reset_saved_out_state ) # resets saved_state and saved_output
        self.error_per_sample= error_per_sample

class SimpleLstmCell(object):
    ''' Single lstm cell
    
    Attributes:
        n_inputs: number of inputs
        n_nodes: nuber of nodes
        afunction: activation function, for the moment it could be tanh and ReLU
        name: name used in all TensorFlow variables' names
        
        saver_dict: saver for the parameters used by the layer
    ''' 
    
    def __init__(self, n_inputs, n_nodes, afunction= 'tanh', name=''): 
        
        self.n_inputs= n_inputs
        self.n_nodes= n_nodes
        self.afunction= afunction
        self.name= name
        
        # 1. Define Trainable Parameters:
        # Input gate: input, previous output, and bias.
        self.ix = tf.Variable(tf.truncated_normal([n_inputs, self.n_nodes], -0.1, 0.1), name=('w_ix'+name))
        self.im = tf.Variable(tf.truncated_normal([self.n_nodes, self.n_nodes], -0.1, 0.1), name=('w_im'+name))
        self.ib = tf.Variable(tf.zeros([1, self.n_nodes]), name=('w_ib'+name))
        # Forget gate: input, previous output, and bias.
        self.fx = tf.Variable(tf.truncated_normal([n_inputs, self.n_nodes], -0.1, 0.1), name=('w_fx'+name))
        self.fm = tf.Variable(tf.truncated_normal([self.n_nodes, self.n_nodes], -0.1, 0.1), name=('w_fm'+name))
        self.fb = tf.Variable(tf.zeros([1, self.n_nodes]), name=('w_fb'+name))
        # Memory cell: input, state and bias.                             
        self.cx = tf.Variable(tf.truncated_normal([n_inputs, self.n_nodes], -0.1, 0.1), name=('w_cx'+name))
        self.cm = tf.Variable(tf.truncated_normal([self.n_nodes, self.n_nodes], -0.1, 0.1), name=('w_cm'+name))
        self.cb = tf.Variable(tf.zeros([1, self.n_nodes]), name=('w_cb'+name))
        # Output gate: input, previous output, and bias.
        self.ox = tf.Variable(tf.truncated_normal([n_inputs, self.n_nodes], -0.1, 0.1), name=('w_ox'+name))
        self.om = tf.Variable(tf.truncated_normal([self.n_nodes, self.n_nodes], -0.1, 0.1), name=('w_om'+name))
        self.ob = tf.Variable(tf.zeros([1, self.n_nodes]), name=('w_ob'+name))
        
        # Define the saver:
        self.saver_dict= dict()
        self.saver_dict['w_ix'+name] =  self.ix
        self.saver_dict['w_im'+name] =  self.im
        self.saver_dict['b_i'+name]  =  self.ib
        self.saver_dict['w_fx'+name] =  self.fx
        self.saver_dict['w_fm'+name] =  self.fm
        self.saver_dict['b_f'+name]  =  self.fb
        self.saver_dict['w_cx'+name] =  self.cx
        self.saver_dict['w_cm'+name] =  self.cm
        self.saver_dict['b_c'+name]  =  self.cb
        self.saver_dict['w_ox'+name] =  self.ox
        self.saver_dict['w_om'+name] =  self.om
        self.saver_dict['b_o'+name]  =  self.ob

    
    # Definition of the cell computation.
    def evaluate(self, i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        
        if self.afunction == 'tanh':
            input_gate =  tf.sigmoid(tf.matmul(i, self.ix) + tf.matmul(o, self.im) + self.ib)
            forget_gate = tf.sigmoid(tf.matmul(i, self.fx) + tf.matmul(o, self.fm) + self.fb)
            update = tf.matmul(i, self.cx) + tf.matmul(o, self.cm) + self.cb
            state = forget_gate * state + input_gate * tf.tanh(update)
            output_gate = tf.sigmoid(tf.matmul(i, self.ox) + tf.matmul(o, self.om) + self.ob)
            return output_gate * tf.tanh(state), state
        
        if self.afunction == 'ReLU':
            input_gate =  tf.sigmoid(tf.matmul(i, self.ix) + tf.matmul(o, self.im) + self.ib)
            forget_gate = tf.sigmoid(tf.matmul(i, self.fx) + tf.matmul(o, self.fm) + self.fb)
            update = tf.matmul(i, self.cx) + tf.matmul(o, self.cm) + self.cb
            state = forget_gate * state + input_gate * tf.nn.relu(update)
            output_gate = tf.sigmoid(tf.matmul(i, self.ox) + tf.matmul(o, self.om) + self.ob)
            return output_gate * tf.nn.relu(state), state
        

class AlexLstmCell(object):
    ''' Single lstm cell defined as in: "Generating Sequences with Recurrent Neural Networks", Alex Graves, 2014
    
    Attributes:
        n_inputs: number of inputs
        n_nodes: number of nodes
        afunction: activation function, for the moment it could be tanh and ReLU, TODO: change it to any function 
        name: name used in all TensorFlow variables' names
    ''' 
    
    def __init__(self, n_inputs, n_nodes, afunction= 'tanh', name=''): 
        
        self.n_inputs= n_inputs
        self.n_nodes= n_nodes
        self.afunction= afunction
        self.name= name
        
        # 1. Define Trainable Parameters:
        # Note: w_ci, w_cf, wc_o are a diagonal matrices, therefore we only reserve space for the diagonal, and the product with the cell state is done as a element-by-element product instead of a matrix product
        # Input gate: input, previous state, previous hidden output and bias.
        self.w_xi = tf.Variable(tf.truncated_normal([n_inputs, n_nodes], -0.1, 0.1), name=('w_xi'+name))
        self.w_hi = tf.Variable(tf.truncated_normal([n_nodes, n_nodes], -0.1, 0.1), name=('w_hi'+name))
        self.w_ci = tf.Variable(tf.truncated_normal([n_nodes], -0.1, 0.1), name=('w_ci'+name)) 
        self.b_i = tf.Variable(tf.zeros([1, n_nodes]), name=('b_i'+name))
        # Forget gate: input, previous state, and bias.
        self.w_xf = tf.Variable(tf.truncated_normal([n_inputs, n_nodes], -0.1, 0.1), name=('w_xf'+name))
        self.w_hf = tf.Variable(tf.truncated_normal([n_nodes, n_nodes], -0.1, 0.1), name=('w_hi'+name))
        self.w_cf = tf.Variable(tf.truncated_normal([n_nodes], -0.1, 0.1), name=('w_cf'+name))
        self.b_f = tf.Variable(tf.zeros([1, n_nodes]), name=('b_f'+name))
        # Memory cell: input, state and bias.                             
        self.w_xc = tf.Variable(tf.truncated_normal([n_inputs, n_nodes], -0.1, 0.1), name=('w_xc'+name))
        self.w_hc = tf.Variable(tf.truncated_normal([n_nodes, n_nodes], -0.1, 0.1), name=('w_hc'+name))
        self.b_c = tf.Variable(tf.zeros([1, n_nodes]), name=('b_c'+name))
        # Output gate: input, previous output, and bias.
        self.w_xo = tf.Variable(tf.truncated_normal([n_inputs, n_nodes], -0.1, 0.1), name=('w_xo'+name))
        self.w_ho = tf.Variable(tf.truncated_normal([n_nodes, n_nodes], -0.1, 0.1), name=('w_ho'+name))
        self.w_co = tf.Variable(tf.truncated_normal([n_nodes], -0.1, 0.1), name=('w_co'+name))
        self.b_o = tf.Variable(tf.zeros([1, n_nodes]), name=('b_o'+name))
        
        # Define the saver:
        self.saver_dict= dict()
        self.saver_dict['w_xi'+name] =  self.w_xi
        self.saver_dict['w_hi'+name] =  self.w_hi
        self.saver_dict['w_ci'+name] =  self.w_ci
        self.saver_dict['b_i'+name]  =  self.b_i
        self.saver_dict['w_xf'+name] =  self.w_xf
        self.saver_dict['w_hf'+name] =  self.w_hf
        self.saver_dict['w_cf'+name] =  self.w_cf
        self.saver_dict['b_f'+name]  =  self.b_f
        self.saver_dict['w_xc'+name] =  self.w_xc
        self.saver_dict['w_hc'+name] =  self.w_hc
        self.saver_dict['b_c'+name]  =  self.b_c
        self.saver_dict['w_xo'+name] =  self.w_xo
        self.saver_dict['w_ho'+name] =  self.w_ho
        self.saver_dict['w_co'+name] =  self.w_co
        self.saver_dict['b_o'+name]  =  self.b_o
        
    def evaluate(self, i, h, state):
        """Create a LSTM cell. 
        i: inputs, on the paper (Alex Graves) the notation is x_t
        h: hidden output on the previous time step, on the paper is h_t-1
        state: state on the previous time step, in the paper is c_t-1
        """
                        
        if self.afunction == 'tanh':
            i_t =  tf.sigmoid(tf.matmul(i, self.w_xi) + tf.matmul(h, self.w_hi) + state*self.w_ci + self.b_i) #input gate
            f_t = tf.sigmoid(tf.matmul(i, self.w_xf) + tf.matmul(h, self.w_hf) + state*self.w_cf + self.b_f)  #forget gate
            update = tf.matmul(i, self.w_xc) + tf.matmul(h, self.w_hc) + self.b_c           #state update
            state = f_t * state + i_t * tf.tanh(update)
            o_t = tf.sigmoid(tf.matmul(i, self.w_xo) + tf.matmul(h, self.w_ho) + state*self.w_co + self.b_o)
            return o_t * tf.tanh(state), state
        
class AlexLstmCellOptimized(object):
    ''' Single lstm cell defined as in: "Generating Sequences with Recurrent Neural Networks", Alex Graves, 2014
    
    Attributes:
        n_inputs: number of inputs
        n_nodes: number of nodes
        afunction: activation function, for the moment it could be tanh and ReLU, TODO: change it to any function 
        name: name used in all TensorFlow variables' names
    ''' 
    
    def __init__(self, n_inputs, n_nodes, afunction= 'tanh', name=''): 
        
        self.n_inputs= n_inputs
        self.n_nodes= n_nodes
        self.afunction= afunction
        self.name= name
        
        # 1. Define Trainable Parameters:
        # Note: w_ci, w_cf, wc_o are a diagonal matrices, therefore we only reserve space for the diagonal, and the product with the cell state is done as a element-by-element product instead of a matrix product
        
        self.w_x = tf.Variable(tf.truncated_normal([n_inputs, 4*n_nodes], -0.1, 0.1), name=('w_xi'+name))
        self.w_h = tf.Variable(tf.truncated_normal([n_nodes, 4*n_nodes], -0.1, 0.1), name=('w_hi'+name))
        #self.w_cprev = tf.Variable(tf.truncated_normal([n_nodes,2], -0.1, 0.1), name=('w_ci'+name)) 
        #self.w_c = tf.Variable(tf.truncated_normal([n_nodes], -0.1, 0.1), name=('w_ci'+name)) 
        
        # Input gate
        self.w_ci = tf.Variable(tf.truncated_normal([n_nodes], -0.1, 0.1), name=('w_ci'+name)) 
        self.b_i = tf.Variable(tf.zeros([1, n_nodes]), name=('b_i'+name))
        # Forget gate: input, previous state, and bias.
        self.w_cf = tf.Variable(tf.truncated_normal([n_nodes], -0.1, 0.1), name=('w_cf'+name))
        self.b_f = tf.Variable(tf.zeros([1, n_nodes]), name=('b_f'+name))
        # Memory cell: input, state and bias.                            
        self.b_c = tf.Variable(tf.zeros([1, n_nodes]), name=('b_c'+name))
        # Output gate: input, previous output, and bias.
        self.w_co = tf.Variable(tf.truncated_normal([n_nodes], -0.1, 0.1), name=('w_co'+name))
        self.b_o = tf.Variable(tf.zeros([1, n_nodes]), name=('b_o'+name))
        
        # Define the saver:
        self.saver_dict= dict()
        self.saver_dict['w_x'+name] =  self.w_x
        self.saver_dict['w_h'+name] =  self.w_h
        
        self.saver_dict['w_ci'+name] =  self.w_ci
        self.saver_dict['w_cf'+name] =  self.w_cf
        self.saver_dict['w_co'+name] =  self.w_co
        
        self.saver_dict['b_i'+name]  =  self.b_i
        self.saver_dict['b_f'+name]  =  self.b_f
        self.saver_dict['b_c'+name]  =  self.b_c
        self.saver_dict['b_o'+name]  =  self.b_o
        
    def evaluate(self, i, h, state):
        """Create a LSTM cell. 
        i: inputs, on the paper (Alex Graves) the notation is x_t
        h: hidden output on the previous time step, on the paper is h_t-1
        state: state on the previous time step, in the paper is c_t-1
        """
        
        if self.afunction == 'tanh':
            # x transformation
            aux_x= tf.matmul(i, self.w_x)
            ix, fx, cx, ox = tf.split(1, 4, aux_x)
            
            # h transformation
            aux_h= tf.matmul(h, self.w_h)
            ih, fh, ch, oh = tf.split(1, 4, aux_h)
                        
            # calculate gate values and output
            i_t = tf.sigmoid( ix + ih + state*self.w_ci + self.b_i) #input gate
            f_t = tf.sigmoid( fx + fh + state*self.w_cf + self.b_f)  #forget gate
            update = cx + ch + self.b_c           #state update
            state = f_t * state + i_t * tf.tanh(update)
            o_t = tf.sigmoid( ox + oh + state*self.w_co + self.b_o)
            return o_t * tf.tanh(state), state
        
        
        
class LstmNet(object):
    ''' network with multiple lstm cells:
    
    inputs_list -> [lstm_cell_1] -> [lstm_cell_2] -> ... -> [lstm_cell_N] -> [linear] -> outputs_list
    
    Attributes:
        n_inputs: number of inputs
        n_nodes: list with the number of nodes on each layer
        n_outputs: number of outputs
        n_extra: list with the number of extra inputs per layer (including first layer)
        n_layers: number of layers
        name: name used in all TensorFlow variables' names
        
        w: weights for final linear transformation
        b: bias for final linear transformation
        
    ''' 
    
    def __init__(self, n_inputs, n_nodes, n_outputs, n_extra=0, 
                 afunction= 'tanh', 
                 LstmCell= SimpleLstmCell ,
                 OutLayer= AffineLayer, 
                 name=''): 
        ''' Define the variables that will represent the learning parameters of the network'''
        # setup inputs and outputs of the network
        self.n_inputs= n_inputs
        self.n_nodes= n_nodes
        self.name= name
        
        self.n_layers= len(n_nodes)
        
        # check n_extra values
        if isinstance(n_extra, list):
            self.n_extra= n_extra
            if len(self.n_extra) != self.n_layers+1:
                raise ValueError('n_extra list must have a length equal to the number of hidden layers plus the output layer')
                
            if (self.n_extra[0] != 0):
                raise ValueError('n_extra[0] must be zero at the moment (no extra inputs for first layer)')
            
        else:
            self.n_extra= [n_extra for i in range(n_layers+1)] # all hidden cells plus the output layer
        
                        
        # Create each one of the cells for each layer
        self.cell_list= list()
        
        self.cell_list.append( LstmCell(n_inputs+self.n_extra[0], n_nodes[0], afunction, name= name+'_L0') )
        
        for l in range(1,self.n_layers):
            self.cell_list.append( LstmCell(n_nodes[l-1]+self.n_extra[l], n_nodes[l], afunction, name= name+'_L'+str(l) ) )
                
        # Final output weights and biases.
        if OutLayer is not None:
            self.n_outputs= n_outputs
            self.out_layer= OutLayer( n_nodes[-1]+self.n_extra[-1], n_outputs)
        else:
            self.out_layer = None
            
        # Saver
        saver_dict= dict()
        for l in range(self.n_layers):
            saver_dict.update( self.cell_list[l].saver_dict )
            
        if self.out_layer is not None:
            saver_dict.update( self.out_layer.saver_dict )
        
        self.saver= tf.train.Saver(saver_dict)
    
    def get_extra_inputs(self, i, h_list, state_list):
        ''' Gets extra inputs for current layer 
        This function should be overwrited by the user if he wants to introduce aditional inputs to the neural network
        i: input to the network
        h_list: list with the hidden outputs of each layer up to the current layer
        state_list: list with the value of the internal state for each cell
        '''
        return None
    
    def evaluate_final_output(self, outputs_list, inputs_list, h_list ):
        ''' Calculates the final output of the neural network, usually it is just a linear transformation
        
        outputs_list: list with the outputs from the last lstm cell
        inputs_list: list of inputs to the network
        h_list: list with all hidden outputs from all the cells, Note: h_list includes outputs_list
        '''
        return self.out_layer.evaluate(tf.concat(0, outputs_list))
        
        
    
    def lstm_net_step(self, i, o_prev, state_prev, drop_prob_list):
        """  Calculates the entire set of next states and outputs for input i
        
        Each layer has its state and output, therefore state and output are lists with
        the state and output for each layer
        i: input to the network
        o_prev: list of hidden outputs in the previous time step
        state_prev: list with the states in the previous time step
        drop_prob_list: list with the dropout placeholders
        """
        out_list= list()
        state_list= list()
        
        # first layer
        if self.n_extra[0]>0:   
            # handle extra inputs
            i= tf.concat(1, [i, self.get_extra_inputs(i, out_list, state_list)]) 
        
        if drop_prob_list[0] is not None:        
            output, state = self.cell_list[0].evaluate(tf.nn.dropout(i, drop_prob_list[0]), o_prev[0], state_prev[0])
        else:
            output, state = self.cell_list[0].evaluate(i, o_prev[0], state_prev[0])
        out_list.append(output)
        state_list.append(state)
        
        # all following layers
        for l in range(1,self.n_layers):
            if self.n_extra[l]>0: 
                # handle extra inputs
                output = tf.concat(1, [output, self.get_extra_inputs(i, out_list, state_list)]) 
            if drop_prob_list[l] is not None:        
                output, state = self.cell_list[l].evaluate(tf.nn.dropout(output, drop_prob_list[l]), o_prev[l], state_prev[l])
            else:
                output, state = self.cell_list[l].evaluate(output, o_prev[l], state_prev[l])
                
            out_list.append(output)
            state_list.append(state)
        
        return out_list, state_list
    
        
    # Definition of the cell computation.
    def unrolling_setup(self, batch_size, num_unrollings, 
                        inputs_list= None, 
                        labels_list= None,
                        drop_prob_list=None,  
                        saved_output= None, saved_state= None,
                        deps_list= None,
                        calculate_loss= True,
                        reset_between_unrollings= False,
                        name= ''
                       ):
        """Unrolls the lstm network.

        Creates an unrolling of the network:
        
        inputs_list -> [lstm_cell_1] -> [lstm_cell_1] -> ... -> [lstm_cell_N] -> [linear] -> outputs_list
                             ^                ^           ^           ^                    
                             |                |           |           |                    
        inputs_list -> [lstm_cell_1] -> [lstm_cell_1] -> ... -> [lstm_cell_N] -> [linear] -> outputs_list
                             ^                ^           ^           ^                    
                             |                |           |           |                    
        inputs_list -> [lstm_cell_1] -> [lstm_cell_1] -> ... -> [lstm_cell_N] -> [linear] -> outputs_list

        Args:
            batch_size:
            num_unrollings:
            inputs_list: 
            labels_list: user can optionally provide its own labels placeholders
            drop_prob_list: 
            saved_output: user can optionally provide its own initialization for the output tensor
            saved_state: user can optionally provide its own initialization for the state tensor
            deps_list: list of dependencies to be runned before calculating final output y and loss
            reset_between_unrollings: reset the saved state and output between unrolling calls 

        Returns:
            A class that defines the important variables for using the unrolling:
        """
        
        # 1. Saved output and state from previous unrollings
        create_feedback= False
        if saved_output is None:
            saved_output = list()
            saved_state = list()
            for l in range(self.n_layers):
                saved_output.append( tf.Variable(tf.zeros([batch_size, self.n_nodes[l]]), trainable=False, 
                                                 name= 'saved_output_L'+str(l)+self.name+name) )
                saved_state.append( tf.Variable(tf.zeros([batch_size, self.n_nodes[l]]), trainable=False,
                                                name= 'saved_state_L'+str(l)+self.name+name) )
            create_feedback= True
                
        # 2. Input data. 
        if inputs_list is None:
            inputs_list = list()
            for iaux in range(num_unrollings):
                inputs_list.append(tf.placeholder(tf.float32, shape=[batch_size, self.n_inputs],
                                                  name= 'inputs_list'+str(iaux)+self.name+name))
                       
        # 3. Unrolled LSTM loop. 
        outputs_list = list()  # list with the hidden output of the last cell
        h_list = list()        # list with all hidden outputs of the network, including outputs_list
        output= saved_output 
        state = saved_state
        #print(state[1]) DELETE
        for i in inputs_list:             
            #output, state = self.lstm_cell(i, output, state) # we introduce dropout here
            output, state = self.lstm_net_step(i, output, state, drop_prob_list)
            
            h_list.append(output)
            
            if drop_prob_list[-1] is not None:
                outputs_list.append( tf.nn.dropout( output[-1], drop_prob_list[-1]) )
            else:
                outputs_list.append(output[-1])
        
        
        # Create a list that assigns state and output to saved state and output
        if create_feedback:
            assign_saved_out_state= list()
            for l in range(self.n_layers):
                assign_saved_out_state.append(saved_output[l].assign( output[l] ))
                assign_saved_out_state.append(saved_state[l].assign( state[l] ))
        else:
            assign_saved_out_state= None
            
        # Create a list to reset saved_state and saved_output
        reset_output_state= list()
        for l in range(self.n_layers):
            reset_output_state.append(saved_output[l].assign( tf.zeros([batch_size, self.n_nodes[l]]) ))
            reset_output_state.append(saved_state[l].assign(  tf.zeros([batch_size, self.n_nodes[l]]) ))
        
        # 4. Final output
        if calculate_loss and (self.out_layer is not None):
            # First create a place holder for labels to be able to calculate the loss
            if labels_list is None:
                labels_list = list()
                for _ in range(num_unrollings): 
                    labels_list.append(tf.placeholder(tf.float32, shape=[batch_size, self.n_outputs]))
                
            if deps_list is None:
                if reset_between_unrollings:
                    with tf.control_dependencies(reset_output_state):
                        #y = tf.nn.xw_plus_b(tf.concat(0, outputs_list), self.w, self.b)
                        y = self.evaluate_final_output(outputs_list, inputs_list, h_list )
                        # For regression:
                        #error_per_sample= y - tf.concat(0, labels_list)   
                        #loss = tf.nn.l2_loss( error_per_sample )
                        # For clasification:
                        error_per_sample = tf.nn.softmax_cross_entropy_with_logits(y, tf.concat(0, labels_list))
                        loss = tf.reduce_mean(error_per_sample)
                else:
                    with tf.control_dependencies(assign_saved_out_state):
                        #y = tf.nn.xw_plus_b(tf.concat(0, outputs_list), self.w, self.b)
                        y = self.evaluate_final_output(outputs_list, inputs_list, h_list )
                        # For regression:
                        #error_per_sample= y - tf.concat(0, labels_list)   
                        #loss = tf.nn.l2_loss( error_per_sample )
                        # For clasification:
                        error_per_sample = tf.nn.softmax_cross_entropy_with_logits(y, tf.concat(0, labels_list))
                        loss = tf.reduce_mean(error_per_sample)
            else:
                with tf.control_dependencies(deps_list):
                    #y = tf.nn.xw_plus_b(tf.concat(0, outputs_list), self.w, self.b)
                    y = self.evaluate_final_output(outputs_list, inputs_list, h_list )
                    # For regression:
                    #error_per_sample= y - tf.concat(0, labels_list)
                    #loss = tf.nn.l2_loss( error_per_sample )
                    # For clasification:
                    error_per_sample = tf.nn.softmax_cross_entropy_with_logits(y, tf.concat(0, labels_list))
                    loss = tf.reduce_mean(error_per_sample)
                    
            # 5. Return
            return LstmLayer( y, inputs_list, 
                              output, state, labels_list, 
                              saved_output, saved_state, 
                              assign_saved_out_state, reset_output_state,
                              error_per_sample), loss 
        
        else:    
            #if deps_list is None:
            return LstmLayer( None, inputs_list, 
                              output, state, None, 
                              saved_output, saved_state, 
                              assign_saved_out_state, reset_output_state, 
                              None), None 
                    
        
class AlexLstmNet(LstmNet):
    
    def __init__(self, n_inputs, n_nodes, n_outputs, n_extra=0, 
                 afunction= 'tanh', 
                 LstmCell= SimpleLstmCell ,
                 OutLayer= AffineLayer, 
                 name=''): 
        
        # check n_extra values
        if isinstance(n_extra, list):
            self.n_extra= n_extra
            if len(self.n_extra) != self.n_layers+1:
                raise ValueError('n_extra list must have a length equal to the number of hidden layers plus the output layer')
                
            if (self.n_extra[0] != 0):
                raise ValueError('n_extra[0] must be zero at the moment (no extra inputs for first layer)')
            
        else:
            if len(num_nodes)>1:
                # for all hidden cells plus the output layer:
                n_extra= [num_inputs for i in range(len(num_nodes)+1)] 
                n_extra[0]= 0
                n_extra[-1]= sum(num_nodes) - num_nodes[-1]
            else:
                n_extra= [0,0]
                
        
        # call init form superclass LstmNet
        super().__init__(n_inputs, n_nodes, n_outputs, n_extra, afunction, LstmCell, OutLayer, name)
        
        
    def get_extra_inputs(self, i, h_list, state_list):
        return i
    
    def evaluate_final_output(self, outputs_list, inputs_list, h_list ):
        ''' Calculates the final output of the neural network, usually it is just a linear transformation
        
        outputs_list: list with the outputs from the last lstm cell
        inputs_list: list of inputs to the network
        h_list: list with all hidden outputs from all the cells
        '''
        ''''''
        all_hidden = list()
        
        for t in h_list: # go trough each time step
            all_hidden.append( tf.concat(1,t) )
        
        return self.out_layer.evaluate(tf.concat(0, all_hidden))  
        
        
class AlexLstmNet_MemOpt(LstmNet): #TODO
    
    def __init__(self, n_inputs, n_nodes, n_outputs, n_extra=0, 
                 afunction= 'tanh', 
                 LstmCell= SimpleLstmCell ,
                 OutLayer= AffineLayer, 
                 name=''): 
        
        # check n_extra values
        if isinstance(n_extra, list):
            self.n_extra= n_extra
            if len(self.n_extra) != self.n_layers+1:
                raise ValueError('n_extra list must have a length equal to the number of hidden layers plus the output layer')
                
            if (self.n_extra[0] != 0):
                raise ValueError('n_extra[0] must be zero at the moment (no extra inputs for first layer)')
            
        else:
            if len(num_nodes)>1:
                # for all hidden cells plus the output layer:
                n_extra= [num_inputs for i in range(len(num_nodes)+1)] 
                n_extra[0]= 0
                n_extra[-1]= sum(num_nodes) - num_nodes[-1]
            else:
                n_extra= [0,0]
                
        
        # call init form superclass LstmNet
        super().__init__(n_inputs, n_nodes, n_outputs, n_extra, afunction, LstmCell, OutLayer, name)
        
        
    def get_extra_inputs(self, i, h_list, state_list):
        return i
    
    def evaluate_final_output(self, outputs_list, inputs_list, h_list ):
        ''' Calculates the final output of the neural network, usually it is just a linear transformation
        
        outputs_list: list with the outputs from the last lstm cell
        inputs_list: list of inputs to the network
        h_list: list with all hidden outputs from all the cells
        '''
        ''''''
        all_hidden = list()
        
        for t in h_list: # go trough each time step
            all_hidden.append( tf.concat(1,t) )
        
        y_list = list()
        for t in all_hidden:
            y_list.append( self.out_layer.evaluate( t ) )
            
        return tf.concat(0, y_list)  
        
