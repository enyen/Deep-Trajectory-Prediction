from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf
from load_data import *

# ((W - F + 2P) / S) + 1
h1_len, h2_len, h3_len, hl_len = 16, 32, 64, 128
out_len = 20

savedModel = 'model1'
restoreModel = True
minibatch, learnrate, dropout, iteration = 64, 1e-4, 0.7, 100


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # read data
    dataset = read_data_sets(out_len)

    # layer input
    x1 = tf.placeholder(tf.float32, [None, 10304])
    x1_ = tf.reshape(x1, [-1, 112, 92, 1])
    keep_prob = tf.placeholder(tf.float32)

    # layer 1 conv
    w1 = weight_variable([7, 7, 1, h1_len])
    b1 = bias_variable([h1_len])
    x2_ = tf.nn.relu(tf.nn.conv2d(x1_, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
    x2 = tf.nn.max_pool(x2_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # layer 2 conv
    w2 = weight_variable([5, 5, h1_len, h2_len])
    b2 = bias_variable([h2_len])
    x3_ = tf.nn.relu(tf.nn.conv2d(x2, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
    x3 = tf.nn.max_pool(x3_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # layer 3 conv
    w3 = weight_variable([3, 3, h2_len, h3_len])
    b3 = bias_variable([h3_len])
    x4_ = tf.nn.relu(tf.nn.conv2d(x3, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
    x4 = tf.nn.max_pool(x4_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # layer 4 fc
    w4 = weight_variable([12 * 14 * h3_len, hl_len])
    b4 = bias_variable([hl_len])
    xl_ = tf.nn.relu(tf.matmul(tf.reshape(x4, [-1, 12 * 14 * h3_len]), w4) + b4)
    xl = tf.nn.dropout(xl_, keep_prob=keep_prob)

    # layer output fc
    wl = weight_variable([hl_len, out_len])
    bl = bias_variable([out_len])
    yl = tf.matmul(xl, wl) + bl

    # loss
    y_ = tf.placeholder(tf.float32, [None, out_len])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yl, y_))
    learn_rate = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cross_entropy)

    # eval
    test_eval = tf.exp(-cross_entropy)

    # variable init
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    if restoreModel:
        saver.restore(sess, savedModel)

        # b_xs, b_ys = dataset.train.next_batch(1)
        # img = tf.reshape(b_xs, [-1, 112, 92, 1])
        # plt.imshow(np.reshape(img.eval(), [112,92]))
        # plt.show()
        # return

        # weight1 = np.reshape(w1.eval(), [7, 7, 16])
        # for i in range(0,16):
        #     plt.subplot(4, 4, i+1)
        #     plt.imshow(weight1[:, :, i], cmap="Greys")
        # plt.show()
        # return
    else:
        tf.initialize_all_variables().run()

    # Train
    performances = []
    i = 0
    while i < iteration:
        # training batch
        batch_xs, batch_ys = dataset.train.next_batch(minibatch)
        train_step.run(feed_dict={x1: batch_xs, y_: batch_ys, keep_prob: dropout, learn_rate: learnrate})

        # performance evaluation
        if i % 1 == 0 and i != dataset.train.epochs_completed:
            train_loss = cross_entropy.eval(feed_dict={x1: batch_xs, y_: batch_ys, keep_prob: 1.0})
            test_x, test_y = dataset.test.next_batch(minibatch)
            test_loss = cross_entropy.eval(feed_dict={x1: test_x, y_: test_y, keep_prob: 1.0})
            print("epoch %d, t_accu %g, v_accu %g" % (i, train_loss, test_loss))
            performances.append([train_loss, test_loss])

        # variables storing
        if (i % 10 == 0) and (i != 0) and i != dataset.train.epochs_completed:
            saver.save(sess, savedModel)
            print("model saved.")

        i = dataset.train.epochs_completed

    np.save('performance3_', np.asarray(np.asarray(performances)))

    saver.save(sess, savedModel)
    print("model saved.")


if __name__ == '__main__':
    tf.app.run()

