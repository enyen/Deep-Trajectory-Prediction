from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
random.seed()


class DataSet(object):

    def __init__(self, data, label):
        self._data = data
        self._label = label
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = data.shape[0]

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._label = self._label[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data[start:end], self._label[start:end]


class Datasets(object):
    def __init__(self, train, test):
        self._train = train
        self._test = test

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test


def read_data_sets(out_len):
    data = np.load('ORL_faces.npz')
    trainX = data['trainX']
    trainY_ = data['trainY']
    testX = data['testX']
    testY_ = data['testY']

    trainY = np.zeros([trainX.shape[0], out_len])
    for i in range(0, trainX.shape[0]):
        trainY[i, trainY_[i]] = 1
    testY = np.zeros([testX.shape[0], out_len])
    for i in range(0, testX.shape[0]):
        testY[i, testY_[i]] = 1

    train = DataSet(trainX, trainY)
    test = DataSet(testX, testY)

    return Datasets(train=train, test=test)
