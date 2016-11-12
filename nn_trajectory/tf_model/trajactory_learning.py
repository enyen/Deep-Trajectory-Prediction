from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
import trajactory_data as input_data
import tensorflow as tf

traj_back = 30
traj_front = 50
order = 2

h1_len = 300
h2_len = 500
h3_len = 700
# out_len = 2*(order+1)
out_len = traj_front * 2
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
    x2 = tf.nn.relu(y1)

    # layer 2
    W2 = tf.Variable(tf.truncated_normal([h1_len, h2_len], stddev=0.1), name="w2")
    b2 = tf.Variable(tf.truncated_normal([h2_len], stddev=0.01), name="b2")
    y2 = tf.matmul(x2, W2) + b2
    x3 = tf.nn.relu(y2)

    # layer 3
    W3 = tf.Variable(tf.truncated_normal([h2_len, h3_len], stddev=0.1), name="w3")
    b3 = tf.Variable(tf.truncated_normal([h3_len], stddev=0.01), name="b3")
    y3 = tf.matmul(x3, W3) + b3
    xl = tf.nn.relu(y3)
    xl_drop = tf.nn.dropout(xl, keep_prob=keep_prob)

    # layer output
    Wl = tf.Variable(tf.truncated_normal([h3_len, out_len], stddev=0.1), name="wl")
    bl = tf.Variable(tf.truncated_normal([out_len], stddev=0.01), name="bl")
    yl = tf.matmul(xl_drop, Wl) + bl

    # loss
    y_ = tf.placeholder(tf.float32, [None, out_len])
    l2loss = tf.reduce_mean(tf.nn.l2_loss(yl - y_))
    train_step = tf.train.AdagradOptimizer(0.005).minimize(l2loss)

    # eval
    predict_loss = tf.reduce_mean(tf.cast(tf.nn.l2_loss(yl - y_), tf.float32))

    # variable init
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    # saver.restore(sess, "model")
    tf.initialize_all_variables().run()

    # Train
    for i in range(100000):
        # performance evaluation
        if i % 1000 == 0:
            train_x, train_y = trajactory.train.next_batch(100)
            train_loss = predict_loss.eval(feed_dict={x1: train_x, y_: train_y, keep_prob: 1.0})
            test_x, test_y = trajactory.test.next_batch(100)
            test_loss = predict_loss.eval(feed_dict={x1: test_x, y_: test_y, keep_prob: 1.0})
            print("step %d, t_loss %g, v_loss %g" % (i, train_loss, test_loss))

        # variables storing
        if (i % 50000 == 0) and (i != 0):
            saver.save(sess, "model")
            print("model saved.")

        # training batch
        batch_xs, batch_ys = trajactory.train.next_batch(100)
        train_step.run(feed_dict={x1: batch_xs, y_: batch_ys, keep_prob: 0.5})

    tf.add_to_collection("yl", yl)
    tf.add_to_collection("x1", x1)
    tf.add_to_collection("keep_prob", keep_prob)
    saver.save(sess, "model")
    print("model saved.")


if __name__ == '__main__':
    tf.app.run()

