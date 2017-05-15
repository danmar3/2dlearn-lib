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
#   Do not redistribute without author's(s') consent
#  
#   Any opinions, findings, and conclusions or recommendations expressed 
#   in this material are those of the author's(s') and do not necessarily 
#   reflect the views of any other entity.
#  
#   ***********************************************************************
#
#   Description: downloads and format the cifar-10 dataset
#                script based from TensorFlow MNIST script: https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/contrib/learn/python/learn/datasets/mnist.py
#
#   ***********************************************************************

import tarfile
import pickle
import os
import numpy as np
import math
import collections
from shutil import copyfile
import tempfile
import urllib


SRC_FILE= 'cifar-10-python.tar.gz'
SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'


Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])



class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 one_hot=False,
                 np_type= np.float32,
                 reshape=True):
        """Construct a DataSet.
        `np_type` can be either `uint8` to leave the input as `[0, 255]`, or 
        `float32` to rescale into `[0, 1]`.
        """
        
        
        if np_type not in (np.uint8, np.float32):
            raise TypeError('Invalid image np type %r, expected uint8 or float32' %
                            np_type)
            
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[3] == 3
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2] * images.shape[3])
        if np_type == np.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        
        
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


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
                
    return filepath

def read_data_sets(train_dir, 
                   one_hot= False, 
                   np_type= np.float32,
                   reshape= True,
                   validation_p= 0.1
                  ):
    
    # 1. download the dataset
    local_file= maybe_download( SRC_FILE, 
                                train_dir, 
                                SOURCE_URL + SRC_FILE)
    
    # 2. extract the dataset
    # decompress
    with tarfile.open(local_file) as in_file:
        in_file.extractall(train_dir)
        
    decomp_path = os.path.join(train_dir, 'cifar-10-batches-py')
    
    # 3. unpickle each training batch
    dataset_x= list()
    dataset_y= list()
    
    for i in range(1,6):
        with open(os.path.join(decomp_path,'data_batch_'+str(i)), 'rb') as f:
            data_i = pickle.load(f, encoding='bytes')
            
            dataset_x.append( data_i[b'data'])
            dataset_y.append( np.array(data_i[b'labels']))

    # 4. create training and validation dataset
    dataset_x= np.concatenate(dataset_x, 0)
    dataset_x= np.reshape( dataset_x, [dataset_x.shape[0], 3, 32, 32 ])
    dataset_x= np.transpose(dataset_x, (0,2,3,1))
    dataset_y= np.concatenate(dataset_y, 0)
    
    validation_size= math.floor(dataset_x.shape[0]*validation_p)
    
    
    valid_x = dataset_x[:validation_size]
    valid_y = dataset_y[:validation_size]
    train_x = dataset_x[validation_size:]
    train_y = dataset_y[validation_size:]
    
    
    # 4. unpickle test dataset
    with open(os.path.join(decomp_path,'test_batch'), 'rb') as f:
        data_i = pickle.load(f, encoding='bytes')
            
        test_x = np.reshape( data_i[b'data'], [data_i[b'data'].shape[0], 3, 32, 32] )
        test_x= np.transpose(test_x, (0,2,3,1))
        test_y = np.array(data_i[b'labels'])
            
    # 5. create datasets
    train = DataSet(train_x, train_y, np_type=np_type, reshape=reshape)
    
    valid = DataSet(valid_x, valid_y, np_type=np_type, reshape=reshape)
    
    test = DataSet(test_x, test_y, np_type=np_type, reshape=reshape)
    
    
    return Datasets(train=train, validation=valid, test=test)

    
    
    
    
    
    
    