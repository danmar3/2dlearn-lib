#  ***********************************************************************
#
#   Description: Creates a generic dataset from a given data
#
#   Created by: Daniel L. Marino (marinodl@vcu.edu)
#    Modern Heuristics Research Group (MHRG)
#    Virginia Commonwealth University (VCU), Richmond, VA
#    http://www.people.vcu.edu/~mmanic/
#
#   ***********************************************************************

import numpy as np
import math


def to_onehot(data_y):
    min_y = np.amin(data_y)
    max_y = np.amax(data_y)
    n_samples = data_y.shape[0]
    out = np.zeros(n_samples, max_y - min_y)
    out[data_y - min_y] = 1.0
    return out


class DataSet(object):
    ''' attributes:
            - _x: samples, [sample_id, sample ...]
            - _y: labels for each sample
            - _epochs_completed
            - _index_in_epoch
            - _n_samples
    '''

    def __init__(self,
                 data_x,
                 data_y=None):
        if data_x is not None:
            self._n_samples = data_x.shape[0]
        else:
            self._n_samples = 0

        self._x = data_x
        self._y = data_y

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._n_samples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if (shuffle == True):
                perm = np.arange(self._n_samples)
                np.random.shuffle(perm)
                self._x = self._x[perm]
                self._y = self._y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._n_samples

        end = self._index_in_epoch
        return self._x[start:end], self._y[start:end]

    def next_batch_x(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._n_samples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if (shuffle == True):
                self.shuffle()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._n_samples

        end = self._index_in_epoch
        return self._x[start:end]

    def update_data(self, data_x, data_y=None):
        self._n_samples = data_x.shape[0]

        self._x = data_x
        self._y = data_y

        self._index_in_epoch = 0

    def shuffle(self):
        perm = np.arange(self._n_samples)
        np.random.shuffle(perm)
        self._x = self._x[perm]
        if self._y is not None:
            self._y = self._y[perm]


class Datasets(object):
    def __init__(self,
                 data_train_x,
                 data_train_y=None,
                 data_valid_x=None,
                 data_valid_y=None,
                 data_test_x=None,
                 data_test_y=None,
                 convert_to_onehot=False):

        # convert dataset to one hot
        if convert_to_onehot is True:
            if (data_train_y is not None):
                data_train_y = to_onehot(data_train_y)

            if (data_valid_y is not None):
                data_valid_y = to_onehot(data_valid_y)

            if (data_test_y is not None):
                data_test_y = to_onehot(data_test_y)

        self.train = DataSet(data_train_x, data_train_y)
        self.valid = DataSet(data_valid_x, data_valid_y)
        self.test = DataSet(data_test_x, data_test_y)
        self._normalized = False
        self._mu = 1.0       # mu used to normalize the dataset
        self._sigma = 1.0    # sigma used to normalize the dataset

    def split_train(self, p_train, p_valid, p_test=0.0, shuffle=True):

        assert (p_train + p_valid + p_test == 1.0), 'percentajes must sum one'
        assert (p_train < 0.0 or p_valid < 0.0 or p_test <
                0.0), 'percentajes must be positive'

        # shuffle the dataset
        if (shuffle == True):
            self.train.shuffle()

        # get number of samples for each dataset
        n_samples = self.train.n_samples()

        n_test = int(math.floor(n_samples * p_test))
        n_valid = int(math.floor(n_samples * p_valid))
        n_train = int(n_samples) - n_test - n_valid

        # update datasets
        data_x = self.train.x
        data_y = self.train.y
        if data_y is not None:
            self.test.update_data(data_x[0:n_test], data_y[0:n_test])
            self.valid.update_data(
                data_x[n_test:n_test + n_valid], data_y[n_test:n_test + n_valid])
            self.train.update_data(
                data_x[n_test + n_valid:], data_y[n_test + n_valid:])

        else:
            self.test.update_data(data_x[0:n_test], None)
            self.valid.update_data(data_x[n_test:n_test + n_valid], None)
            self.train.update_data(data_x[n_test + n_valid:], None)

    def normalize(self, force=False):
        ''' Normalize dataset for having a training dataset with
        zero mean and standard deviation of one.
        @param force: force normalization of the dataset when it has already
                      been normalized
        @return mu, sigma: mean and standard deviation of training
                           dataset
        '''
        tol = 1e-9
        if self._normalized:
            mu = np.mean(self.train._x, 0)
            sigma = np.std(self.train._x, 0)
            # check if mu and sigma of training dataset are the same as
            # those computed in the past
            if ((np.sum(np.abs(mu - self._mu)) < tol) or
                    (np.abs(sigma - self._sigma)) < tol):
                mu = self._mu
                sigma = self._sigma
            elif force:
                self._normalized = False
                print('Training dataset changed, performing normalization')

        if not self._normalized:
            mu = np.mean(self.train._x, 0)
            sigma = np.std(self.train._x, 0)

            self.train._x = (self.train._x - mu) * (1.0 / sigma)

            if self.valid.x is not None:
                self.valid._x = (self.valid._x - mu) / sigma

            if self.test.x is not None:
                self.test._x = (self.test._x - mu) / sigma

            self._mu = mu
            self._sigma = sigma
            self._normalized = True

        return mu, sigma
