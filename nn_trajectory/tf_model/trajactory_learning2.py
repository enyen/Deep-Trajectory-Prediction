from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
import trajactory_data as input_data
import tensorflow as tf

# ((W - F + 2P) / S) + 1
scan_size = 173
traj_back, traj_front, order = 30, 30, 2
h1_len, h2_len, h3_len, hl_len = 64, 128, 256, 512
out_len = traj_front * 2

savedModel = 'saved_scan1/model'
restoreModel = False
minibatch, learnrate, dropout, iteration = 100, 1e-5, 0.5, 20000
regenerateData = False


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # produce data
    if regenerateData:
        input_data.produce_data_scan1(traj_back, traj_front, order)
        print("data produced")
        return

    # read data
    trajactory = input_data.read_data_sets(traj_back, traj_front)

    # layer input
    x1 = tf.placeholder(tf.float32, [None, 1, scan_size, traj_back], name='x1')
    keep_prob = tf.placeholder(tf.float32)

    # layer 1
    w1 = weight_variable([1, 3, traj_back, h1_len])
    b1 = bias_variable([h1_len])
    x2 = tf.nn.relu(tf.nn.conv2d(x1, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)

    # layer 2
    w2 = weight_variable([1, 5, h1_len, h2_len])
    b2 = bias_variable([h2_len])
    x3 = tf.nn.relu(tf.nn.conv2d(x2, w2, strides=[1, 2, 2, 1], padding='VALID') + b2)

    # layer 3
    w3 = weight_variable([1, 5, h2_len, h3_len])
    b3 = bias_variable([h3_len])
    x4_ = tf.nn.relu(tf.nn.conv2d(x3, w3, strides=[1, 2, 2, 1], padding='VALID') + b3)
    x4 = tf.nn.dropout(x4_, keep_prob=keep_prob, name='keep_prob')

    # layer 4
    w4 = weight_variable([1, 3, h3_len, hl_len])
    b4 = bias_variable([hl_len])
    xl_ = tf.nn.relu(tf.nn.conv2d(x4, w4, strides=[1, 2, 2, 1], padding='VALID') + b4)
    xl = tf.nn.dropout(xl_, keep_prob=keep_prob, name='keep_prob')

    # layer output
    wl = weight_variable([20*hl_len, out_len])
    bl = bias_variable([out_len])
    yl = tf.matmul(tf.reshape(xl, [-1, 20*hl_len]), wl) + bl

    # loss
    y_ = tf.placeholder(tf.float32, [None, out_len], name='y_')
    l2loss = tf.reduce_mean(tf.nn.l2_loss(yl - y_))
    learn_rate = tf.placeholder(tf.float32, name='learn_rate')
    train_step = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(l2loss)
    momentum = tf.placeholder(tf.float32, name='momentum')
    train_online = tf.train.MomentumOptimizer(learn_rate, momentum, name='train_online').minimize(l2loss)

    # eval
    predict_loss = tf.reduce_mean(tf.cast(tf.nn.l2_loss(yl - y_), tf.float32))

    # variable init
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    if restoreModel:
        saver.restore(sess, savedModel)
    else:
        tf.initialize_all_variables().run()
    tf.add_to_collection("yl", yl)
    tf.add_to_collection("x1", x1)
    tf.add_to_collection("keep_prob", keep_prob)
    tf.add_to_collection("y_", y_)
    tf.add_to_collection("train_online", train_online)
    tf.add_to_collection("learn_rate", learn_rate)
    tf.add_to_collection("momentum", momentum)

    # Train
    for i in range(iteration):
        # performance evaluation
        if i % 50 == 0:
            train_x, train_y = trajactory.train.next_batch(minibatch)
            train_loss = predict_loss.eval(feed_dict={x1: train_x, y_: train_y, keep_prob: 1.0})
            test_x, test_y = trajactory.test.next_batch(minibatch)
            test_loss = predict_loss.eval(feed_dict={x1: test_x, y_: test_y, keep_prob: 1.0})
            print("step %d, t_loss %g, v_loss %g" % (i, train_loss, test_loss))

        # variables storing
        if (i % 1000 == 0) and (i != 0):
            saver.save(sess, savedModel)
            print("model saved.")

        # training batch
        batch_xs, batch_ys = trajactory.train.next_batch(minibatch)
        train_step.run(feed_dict={x1: batch_xs, y_: batch_ys, keep_prob: dropout, learn_rate: learnrate})

    saver.save(sess, savedModel)
    print("model saved.")


if __name__ == '__main__':
    tf.app.run()

