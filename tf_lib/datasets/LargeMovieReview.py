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
#   Description: downloads and formats the Large Movie Review Dataset (http://ai.stanford.edu/~amaas/data/sentiment/)
#
#   ***********************************************************************

import tarfile
import pickle
import os
import glob
import numpy as np
import math
import collections
from shutil import copyfile
import tempfile
import urllib
from bs4 import BeautifulSoup

#import Batch_generator
import twodlearn.tf_lib.datasets.vocabulary_stem_coding as vc_stem

SRC_FILE= 'aclImdb_v1.tar.gz'
SOURCE_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/'


Datasets = collections.namedtuple('Datasets', ['train', 'valid', 'test'])


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings, vc):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        self.vc= vc
        
        segment = self._text_size // batch_size
        self._cursor = [ offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()
        
    
    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, self.vc.vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, self._text[self._cursor[b]]] = 1.0    # the letter pointed by the cursor is converted to 1-hot encoding
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size  # Here, the cursor is increased
        return batch
    
    @property
    def batch_size(self):
        return self._batch_size
    @batch_size.setter
    def batch_size(self, new_val):
        if self._batch_size != new_val:
            self._batch_size= new_val
            
            segment = self._text_size // self._batch_size
            self._cursor = [ offset * segment for offset in range(self._batch_size)]
            self._last_batch = self._next_batch()
    
    @property
    def num_unrollings(self):
        return self._num_unrollings
    
    @num_unrollings.setter
    def num_unrollings(self, new_val):        
        if self._num_unrollings != new_val:
            self._num_unrollings= new_val
                
            segment = self._text_size // self._batch_size
            self._cursor = [ offset * segment for offset in range(self._batch_size)]
            self._last_batch = self._next_batch()
            
            
    
    def next(self, batch_size= None, num_unrollings= None):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        # 1. handle change of batch_size or num_unrollings
        if batch_size is not None:
            self.batch_size = batch_size
            
        if num_unrollings is not None:
            self.num_unrollings = num_unrollings           
        
        
        # 2. get next batch
        batches = [self._last_batch] # include last batch from previous array
        #batches = list()
        for step in range(self._num_unrollings):
            # each call of _next_batch() increases the cursor by one
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches
    

class DataSet(object):
        
    def __init__(self,
                 data,
                 vc,
                 batch_size,
                 num_unrollings,
                 np_type= np.float32,
                 shuffle= True
                ):
        """Construct a DataSet.
        - text: list, every element corresponds to the text to a different class
        - np_type: can be either `uint8` to leave the input as `[0, 255]`, or 
          `float32` to rescale into `[0, 1]`.
        """
        
        
        if np_type not in (np.uint8, np.float32):
            raise TypeError('Invalid image np type %r, expected uint8 or float32' %
                            np_type)
                
        # define the batch generators
        self._batch_generators= list()
        self._n_classes= len(data)
        for data_i in data:
            self._batch_generators.append( BatchGenerator(data_i, batch_size//self._n_classes, num_unrollings, vc) )
        
        self._shuffle= shuffle
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def batch_generators(self):
        return self._batch_generators
       
    @property
    def n_classes(self):
        return self._n_classes
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def set_batch_and_unrollings(self, batch_size, num_unrollings):
        for batch_i in range(self._n_classes):
            self._batch_generators[batch_i].batch_size= batch_size//self._n_classes
            self._batch_generators[batch_i].num_unrollings= num_unrollings
    
    def next_batch(self, batch_size= None, num_unrollings= None):
        ''' Generates a batch of sentences and its corresponding labels for classification
            Returns:
                X: list of positive and negative sentences, the format is the same that the batches generated from the 
                   positive and negative texts. len(X)= num_unrrolings, X[i].shape: [batch_size x vocabulary_size]
                y: class to which each sentence belong. The format is a vector of size [batch_size x 1]
        '''
        if batch_size is not None:
            batch_size_i = batch_size//self._n_classes
        else:
            batch_size_i = None
        
        xl = list()
        yl = list()
        # get data from batch generators
        for batch_i in range(self._n_classes):
            xl.append( self._batch_generators[batch_i].next(batch_size_i, num_unrollings) )
            yl.append( batch_i*np.ones([xl[-1][0].shape[0], 1]) )
        
        y = np.concatenate(yl)
        batch_size = y.shape[0]
        
        if self._shuffle:           
            rand_ind = np.random.permutation(batch_size) # rand_ind= random shuffling for the sentences
            y = y[rand_ind,:]
        
        x = list()
        for x_ind in range(len(xl[0])):  # the length of batch_pos is the number of unrollings
            if self._shuffle:
                x.append( np.concatenate( [xl_i[x_ind] for xl_i in xl] , 0 )[rand_ind,:] )
            else:
                x.append( np.concatenate( [xl_i[x_ind] for xl_i in xl] , 0 ) )
        
        return x, y


def maybe_download(filename, work_directory, source_url):
    """Download the data from source url, unless it's already here.
    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.
    Returns:
        Path to resulting file.
    """
    
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
            
    filepath = os.path.join(work_directory, filename)
    
    if not os.path.exists(filepath):
        with tempfile.NamedTemporaryFile() as tmpfile:
            temp_file_name = tmpfile.name
            urllib.request.urlretrieve(source_url, temp_file_name)
            copyfile(temp_file_name, filepath)
            
            size = os.path.getsize(filepath)
            print('Successfully downloaded', filename, size, 'bytes.')
    else:
        print('File already downloaded.')
                
    return filepath


def join_files_in_dir(work_dir):
    out_str= ''
    for filename in glob.glob(os.path.join(work_dir, '*.txt')):
        with open(filename, 'r') as file_i:
            out_str +=' '+ file_i.read()
            
    return out_str

def remove_html(string_in):
    soup = BeautifulSoup(string_in, "lxml")
    return soup.get_text()
    

def read_data_sets(data_dir, 
                   batch_size,
                   num_unrollings,
                   np_type= np.float32,
                   validation_p= 0.1,
                   custom_preproc= None,
                   coding='stem',
                   shuffle= True,
                   vc_size = 1000
                  ):
    
    # 1. download the dataset
    local_file= maybe_download( SRC_FILE, 
                                data_dir, 
                                SOURCE_URL + SRC_FILE)
    
    # 2. extract the dataset
    print('Decompressing data file...')
    with tarfile.open(local_file) as in_file:
        in_file.extractall(data_dir)
        
    # 3. join the text files in a single string
    print('Reading training text files...')
    train_dir = os.path.join(data_dir, 'aclImdb/train/')
    train_pos = join_files_in_dir(os.path.join(train_dir, 'pos/'))
    train_neg = join_files_in_dir(os.path.join(train_dir, 'neg/'))
    
    
    print('Reading testing text files...')
    test_dir = os.path.join(data_dir, 'aclImdb/test/')
    test_pos = join_files_in_dir(os.path.join(test_dir, 'pos/'))
    test_neg = join_files_in_dir(os.path.join(test_dir, 'neg/'))
    
    # 4. remove html tags
    print('Parsing text files...')
    train_pos = remove_html(train_pos)
    train_neg = remove_html(train_neg)
    test_pos = remove_html(test_pos)
    test_neg = remove_html(test_neg)
    
    # 5. applying custom preprocessing
    if custom_preproc is not None:
        print('Applying custom preprocessing...')
        train_pos = custom_preproc(train_pos)
        train_neg = custom_preproc(train_neg)
        test_pos = custom_preproc(test_pos)
        test_neg = custom_preproc(test_neg)
    
    # 6. create data with the desired coding
    if coding=='stem':
        # Construct vocabulary
        print("Building vocabulary...")
        
        aux_text= train_pos + train_neg
        print('Size of string to build the dictionary:', len(aux_text))
        aux_text= aux_text.split(' ')
        vc= vc_stem.Vocabulary(aux_text, vc_size)
        
        # Encode the train text  
        print("Encoding texts...")
        train_pos= vc.text2keys(train_pos.split(), ignore_unknown= True)
        train_neg= vc.text2keys(train_neg.split(), ignore_unknown= True)
        
        # Encode the test text
        test_pos= vc.text2keys(test_pos.split(), ignore_unknown= True)
        test_neg= vc.text2keys(test_neg.split(), ignore_unknown= True)
    
    # cut datasets so positive and negative have the same length
    len_train= min(len(train_pos), len(train_neg)) 
    train_pos= train_pos[:len_train]
    train_neg= train_neg[:len_train]
    
    len_test= min(len(test_pos), len(test_neg)) 
    test_pos= test_pos[:len_test]
    test_neg= test_neg[:len_test]
    
    # 7. divide training into train and validation
    valid_size = int(float(len_train)*validation_p)
    valid_pos= train_pos[:valid_size]
    valid_neg= train_neg[:valid_size]
    
    train_pos= train_pos[valid_size:]
    train_neg= train_neg[valid_size:]
    
    # 8. create the datasets         
    print("Creating datasets...")
    train = DataSet([train_neg, train_pos], vc, batch_size, num_unrollings, np_type=np_type, shuffle= shuffle)
    
    valid = DataSet([valid_neg, valid_pos], vc, batch_size, num_unrollings, np_type=np_type, shuffle= shuffle)
    
    test = DataSet([test_neg, test_pos], vc, batch_size, num_unrollings, np_type=np_type, shuffle= shuffle)
        
        
    return Datasets(train=train, valid=valid, test=test), vc
    
    
    
    
    
    