from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from tensorflow.contrib.learn.python.learn.datasets import base
random.seed()


class DataSet(object):

    def __init__(self,
                 data,
                 label,
                 len_back,
                 len_front):
        self._data = data
        self._label = label
        self._back = len_back
        self._front = len_front
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


def read_data_sets(len_back, len_front, test_size=1000):
    data_file = 'traj_data.npy'
    label_file = 'traj_label.npy'
    data = np.load(data_file)
    label = np.load(label_file)

    perm = np.arange(data.shape[0])
    np.random.shuffle(perm)
    data = data[perm]
    label = label[perm]

    train_data = data[:(data.shape[0] - test_size)]
    test_data = data[(data.shape[0] - test_size):]
    train_label = label[:(label.shape[0] - test_size)]
    test_label = label[(label.shape[0] - test_size):]

    train = DataSet(train_data, train_label, len_back, len_front)
    test = DataSet(test_data, test_label, len_back, len_front)
    validate = DataSet(test_data, test_label, len_back, len_front)

    return base.Datasets(train=train, test=test, validation=validate)


def produce_data(len_back, len_front, order):
    local_file = 'trajactory.txt'
    with open(local_file, 'rb') as f:
        raw_data = []
        for line in f:
            raw_data.append(line.split())

    raw_data = np.asarray(np.asarray(raw_data, dtype=np.float32))
    # data_stat = np.zeros([2,3])
    # data_stat[0,:] = np.mean(raw_data, axis=0)
    # data_stat[1,:] = np.std(raw_data, axis=0)
    # raw_data -= data_stat[0,:]
    # raw_data /= data_stat[1,:]

    train_data = []
    train_label = []
    for idx in range(len_back, (raw_data.shape[0]-len_front)):
        data = raw_data[(idx - len_back):(idx + len_front)]

        # data[:, 0] -= data[0, 0]
        train_data.append(np.reshape(data[:len_back], -1))

        X = np.zeros([len_front, order + 1])
        Y = np.zeros([len_front, 2])
        accum = 0
        for i in range(0, len_front):
            accum += data[(len_back + i), 0]
            for j in range(0, order + 1):
                X[i, j] = np.power(accum, j)
            Y[i, 0] = data[(len_back + i), 1]
            Y[i, 1] = data[(len_back + i), 2]
        temp = np.linalg.solve(X.transpose().dot(X), X.transpose())
        params = np.reshape(temp.dot(Y).transpose(), -1)
        # train_label.append(params)

        # X = np.zeros([len_back+len_front, order + 1])
        # Y = np.zeros([len_back+len_front, 2])
        # accum = 0
        # for i in range(0, len_back+len_front):
        #     accum += data[i, 0]
        #     for j in range(0, order + 1):
        #         X[i, j] = np.power(accum, j)
        #     Y[i, 0] = data[i, 1]
        #     Y[i, 1] = data[i, 2]
        # temp = np.linalg.solve(X.transpose().dot(X), X.transpose())
        # params = np.reshape(temp.dot(Y).transpose(), -1)
        # train_label.append(params)

        accum = 0
        delta = np.zeros(2*len_front)
        for i in range(0, len_front):
            accum += 0.1
            delta[i * 2] = params[0] + accum * params[1] + accum**2 * params[2] - data[(len_back-1), 1]
            delta[i*2+1] = params[3] + accum * params[4] + accum**2 * params[5] - data[(len_back-1), 2]
        train_label.append(np.reshape(delta, -1))

    np.save('traj_data', np.asarray(train_data))
    np.save('traj_label', np.asarray(train_label))
    # np.save('traj_stat', data_stat)


# def produce_data(len_back, len_front, order):
#     local_file = 'trajactory.txt'
#     with open(local_file, 'rb') as f:
#         raw_data = []
#         for line in f:
#             raw_data.append(line.split())
#
#     raw_data = np.asarray(np.asarray(raw_data, dtype=np.float32))
#     data_stat = np.zeros([2,3])
#     data_stat[0,:] = np.mean(raw_data, axis=0)
#     data_stat[1,:] = np.std(raw_data, axis=0)
#     raw_data -= data_stat[0,:]
#     raw_data /= data_stat[1,:]
#
#     train_data = []
#     train_label = []
#     for idx in range(len_back, (raw_data.shape[0]-len_front)):
#         data = raw_data[(idx - len_back):(idx + len_front)]
#         train_data.append(np.reshape(data[:len_back], -1))
#         # train_label.append(np.reshape(data[len_back:], -1))
#         train_label.append(np.reshape(data[len_back:, 1:], -1))
#
#     np.save('traj_data', np.asarray(train_data))
#     np.save('traj_label', np.asarray(train_label))
#     np.save('traj_stat', data_stat)
