############################################################################
# This file defines some classes that are usefull for batch generation 
# used for training models
# 
# Created by: Daniel L. Marino (daniellml55@gmail.com)
# Last modified: Jul-25-2016
# 
############################################################################

import numpy as np


def one_hot(labels, n_classes):
    ''' Converts a 1-D list of labels into a one-hot encoding representation'''
    if n_classes==2:
        return np.expand_dims(labels,1)
    else:
        labels_out= np.zeros(labels.shape[0], n_classes)
        for c in range(n_classes):
            labels_out[labels==c,c]= 1

        return labels_out

# ------------------------------------------------- Sequence Batch Generators -------------------------------------------------#
class SeqBatchGenerator(object):
    ''' Generates batches for a single class, considering discontinuous data divided in pages.
        
    For data that is piecewise continuous, each continuous part is called a page. Therefore
    the hole dataset is a list of pages. 
    
    The windows generated from the dataset contain only continuous data.
    
    
    
    '''
    def __init__(self, data, batch_size, window_size, out_format='tensor'):
        '''
        data: list of tensors, each tensor correspond to one page, the pages should be ordered in increasing order of time
        '''
        self.data= data # create a reference to the data
        self.batch_size= batch_size
        self.window_size= window_size
        
        self.page_len= [ p.shape[0] for p in data]
        self.length= sum( self.page_len ) # the length is the total number of elements
        
        spacing= self.length//batch_size # space between cursors
        self.cursor = [ offset * spacing for offset in range(batch_size)] # set the cursor for each batch
        
        self.cumlength= np.cumsum(np.array(self.page_len))
        # get the page to which each cursor is currently pointing
        #self.cursor_page = [len(self.page_len)-len(length_accum[c<length_accum]) for c in cursor] 
        self.cursor_page = [np.argmax(c<self.cumlength) for c in self.cursor] 
        
        # set function to get next batch in the desired format
        if out_format=='tensor':
            self.get_next= self.get_next_tensor       
        elif out_format=='list_windows':
            self.get_next= self.get_next_list
        elif out_format=='list_tensor':
            self.get_next= self.get_next_list_tensor
        
        
    def get_next_list_windows(self):
        # TODO
        return False
    
    def get_next_list_tensor(self, inc= None):
        '''Returns the next batch in a list, where each element of the list
        corresponds to a single sequence. 
        '''
        batch= list()
        # define the increment of the cursor between calls
        if inc is None:
            inc= self.window_size
            
        for b in range(self.batch_size): # each element of the batch has a cursor
            # confirm that the current window stays in the page
            while (self.cursor[b]+self.window_size)>(self.cumlength[self.cursor_page[b]]):
                #self.cursor[b] = (self.cursor[b] + self.window_size )%self.length
                self.cursor[b] = (self.cursor[b] + inc )%self.length
                self.cursor_page[b] = np.argmax(self.cursor[b]<self.cumlength)
            
            # get window for current cursor
            start_idx= self.cursor[b] - (self.cumlength[self.cursor_page[b]] - self.page_len[self.cursor_page[b]])
            batch.append( self.data[ self.cursor_page[b] ][start_idx:(start_idx+self.window_size)] )
            
            # update cursor
            #self.cursor[b] = (self.cursor[b] + self.window_size)%self.length
            self.cursor[b] = (self.cursor[b] + inc)%self.length
            self.cursor_page[b] = np.argmax(self.cursor[b]<self.cumlength)
            
        return batch
        
    def get_next_tensor(self, inc= None):
        return np.stack(self.get_next_list_tensor(inc), axis=0)
    

class SeqMultiClassBatchGenerator(object):
    ''' Generates batches of sequences for multiple classes
    
    Atributes:
        - bg_l: list containing the batch generator for each class
        - data: datset from where the batches are generated. Is structured as data[class][page] after initialization
        - n_classes: number of classes
        - batch_size:
        - window_size:
    '''
    
    def __init__(self, data_in, batch_size, window_size, mrk= None, y= None, out_format='tensor'):
        '''
        At the end of the initialization, data is structured as data[class][page]
        
        data: -list containing tensors from which the batches are generated, each tensor corresponds to a class
              -tensor containing 
        
        mrk: markers used to divide the data into pages
        y: if y is provided, data is not divided by class. y has to be formated as int
        n_classes: number of classes
        bg_l: list of batch generators. Each element is a SeqBatchGenerator corresponding to a batch generator
              for each class.
        
        '''
        self.batch_size= batch_size
        self.window_size= window_size
        
        if (type(data_in) is np.ndarray) and (mrk is not None) and (y is not None):
            self.n_classes= np.amax(y)-np.amin(y) + 1
            
            self.data= [list() for i in range(int(self.n_classes))] # Create a list for each class
            
            # get markers, TODO: add an option to specify how the markers are specified
            aux_mrk= np.squeeze(np.transpose(np.nonzero( np.abs(np.diff(mrk)))))
            aux_mrk= aux_mrk.astype(int)
            #print(aux_mrk, self.n_classes, self.data)
            
            label= y[aux_mrk-1]
            # separate data in classes and pages
            self.data[label[0]].append( data_in[0:(aux_mrk[0]+1)] )
            for m in range(aux_mrk.shape[0]-1):
                self.data[label[m+1]].append( data_in[(aux_mrk[m]+1):(aux_mrk[m+1]+1)] )
            self.data[y[-1]].append( data_in[aux_mrk[m+1]:] )
            
            #Create the list of batch generators for each class
            self.bg_l= list()
            for c in range(len(self.data)):
                self.bg_l.append( SeqBatchGenerator( self.data[c], batch_size, window_size, out_format) )
            
        elif (type(data_in) is list) and (mrk is not None) and (y is None):
            self.n_classes= len(data_in)
            self.data= [list() for i in self.n_clases] # Create a list for each class
            
            #TODO: continue!!!!!!
        elif (type(data_in) is list) and (type(data_in[0]) is list) and (mrk is None) and (y is None):
            self.n_classes= len(data_in)
            self.data= data_in
            
            self.bg_l= list()
            for c in range(len(self.data)):
                self.bg_l.append( SeqBatchGenerator( self.data[c], batch_size, window_size, out_format) )
            
        
    def get_next(self, inc= None):
        
        out= [ bg.get_next() for bg in self.bg_l ]
        labels= np.concatenate([ c*np.ones([t.shape[0]]) for t,c in zip(out,range(len(out))) ])
                
        return np.concatenate(out, axis=0), one_hot(labels, self.n_classes)
        
        
    
        
# ------------------------------------------------- Standard Batch Generators -------------------------------------------------#
class BatchGenerator(object):
    '''Generates batches in form of a matrix, where each row constitutes a single sample'''
    def __init__(self, X, y, batch_size, RandomShuffle= True, task= 'classification'):
        self.X= X
        self.batch_size= batch_size
        
        if task=='classification':
            self.n_classes= np.amax(y)-np.amin(y) + 1
            self.y= one_hot(y, self.n_classes)
        else:
            self.y= y
        
        if RandomShuffle:
            randp= np.random.permutation(self.X.shape[0])
            self.X= self.X[randp,:]
            self.y= self.y[randp,:]
        
        self.cursor= 0
        
    def get_next(self):
        init_i= self.cursor
        end_i= self.cursor + self.batch_size
        self.cursor= (end_i) % (self.X.shape[0] - self.batch_size)
        return self.X[init_i:end_i,:], self.y[init_i:end_i,:]
        













