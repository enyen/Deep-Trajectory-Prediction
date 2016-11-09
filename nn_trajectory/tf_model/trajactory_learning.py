from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Import data
import trajactory_data as input_data
import tensorflow as tf

traj_back = 20
traj_front = 20
order = 2

h1_len = 300
h2_len = 600
out_len = 2*(order+1)
# out_len = traj_front * 3


def main(_):
    # produce data
    # input_data.produce_data(traj_back, traj_front, order)
    # print("data produced")
    # return

    # read data
    trajactory = input_data.read_data_sets(traj_back, traj_front)

    # layer input
    x1 = tf.placeholder(tf.float32, [None, traj_back * 3])
    keep_prob = tf.placeholder(tf.float32)

    # layer 1
    W1 = tf.Variable(tf.truncated_normal([traj_back * 3, h1_len], stddev=0.1), name="w1")
    b1 = tf.Variable(tf.truncated_normal([h1_len], stddev=0.01), name="b1")
    y1 = tf.matmul(x1, W1) + b1
    y1_norm = tf.nn.l2_normalize(y1, 0)
    x2 = tf.nn.relu(y1_norm)
    x2_drop = tf.nn.dropout(x2, keep_prob=keep_prob)

    # layer 2
    W2 = tf.Variable(tf.truncated_normal([h1_len, h2_len], stddev=0.1), name="w2")
    b2 = tf.Variable(tf.truncated_normal([h2_len], stddev=0.01), name="b2")
    y2 = tf.matmul(x2_drop, W2) + b2
    y2_norm = tf.nn.l2_normalize(y2, 0)
    x3 = tf.nn.relu(y2_norm)
    x3_drop = tf.nn.dropout(x3, keep_prob=keep_prob)

    # layer output
    W3 = tf.Variable(tf.truncated_normal([h2_len, out_len], stddev=0.1), name="w3")
    b3 = tf.Variable(tf.truncated_normal([out_len], stddev=0.01), name="b3")
    y3 = tf.matmul(x3_drop, W3) + b3

    # loss
    y_ = tf.placeholder(tf.float32, [None, out_len])
    l2loss = tf.reduce_mean(tf.nn.l2_loss(y3 - y_))
    train_step = tf.train.AdagradOptimizer(0.02).minimize(l2loss)

    # eval
    predict_loss = tf.reduce_mean(tf.cast(tf.nn.l2_loss(y3 - y_), tf.float32))

    # variable init
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    # saver.restore(sess, "model")
    tf.initialize_all_variables().run()

    # Train
    for i in range(1000):
        # performance evaluation
        if i % 1000 == 0:
            train_x, train_y = trajactory.train.next_batch(200)
            train_loss = predict_loss.eval(feed_dict={x1: train_x, y_: train_y, keep_prob: 1.0})
            test_x, test_y = trajactory.test.next_batch(200)
            test_loss = predict_loss.eval(feed_dict={x1: test_x, y_: test_y, keep_prob: 1.0})
            print("step %d, t_loss %g, v_loss %g" % (i, train_loss, test_loss))

        # variables storing
        if (i % 50000 == 0) and (i != 0):
            saver.save(sess, "model")
            print("model saved.")

        # training batch
        batch_xs, batch_ys = trajactory.train.next_batch(100)
        train_step.run(feed_dict={x1: batch_xs, y_: batch_ys, keep_prob: 0.5})

    tf.add_to_collection("y3", y3)
    saver.save(sess, "model")
    # tf.train.export_meta_graph("model.meta", collection_list=["y3"])
    print("model saved.")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='/tmp/data',
    #                     help='Directory for storing data')
    # FLAGS = parser.parse_args()
    tf.app.run()

