from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
import trajactory_data as input_data
import tensorflow as tf

traj_back, traj_front, order = 30, 30, 2
h1_len, h2_len, h3_len = 200, 300, 400
out_len = 2*(order+1)
# out_len = traj_front * 2
# out_len = traj_front * 3

savedModel = 'saved1/model'
restoreModel = True
minibatch, learnrate, dropout, iteration = 100, 1e-7, 0.5, 200000
regenerateData = False


def main(_):
    # produce data
    if regenerateData:
        input_data.produce_data1(traj_back, traj_front, order)
        print("data produced")
        return

    # read data
    trajactory = input_data.read_data_sets(traj_back, traj_front)

    # layer input
    x1 = tf.placeholder(tf.float32, [None, traj_back * 3], name='x1')
    keep_prob = tf.placeholder(tf.float32)

    # layer 1
    W1 = tf.Variable(tf.truncated_normal([traj_back * 3, h1_len], stddev=0.1), name="w1")
    b1 = tf.Variable(tf.truncated_normal([h1_len], mean=0.05, stddev=0.05), name="b1")
    y1 = tf.matmul(x1, W1) + b1
    # x2 = tf.nn.relu(y1)
    x2 = tf.tanh(y1)
    # x2_drop = tf.nn.dropout(x2, keep_prob=keep_prob)

    # layer 2
    W2 = tf.Variable(tf.truncated_normal([h1_len, h2_len], stddev=0.1), name="w2")
    b2 = tf.Variable(tf.truncated_normal([h2_len], mean=0.05, stddev=0.05), name="b2")
    y2 = tf.matmul(x2, W2) + b2
    # x3 = tf.nn.relu(y2)
    x3 = tf.tanh(y2)
    # x3_drop = tf.nn.dropout(x3, keep_prob=keep_prob)

    # layer 3
    W3 = tf.Variable(tf.truncated_normal([h2_len, h3_len], stddev=0.1), name="w3")
    b3 = tf.Variable(tf.truncated_normal([h3_len], mean=0.05, stddev=0.05), name="b3")
    y3 = tf.matmul(x3, W3) + b3
    # xl = tf.nn.relu(y3)
    xl = tf.tanh(y3)
    xl_drop = tf.nn.dropout(xl, keep_prob=keep_prob, name='keep_prob')

    # layer output
    Wl = tf.Variable(tf.truncated_normal([h3_len, out_len], stddev=0.1), name="wl")
    bl = tf.Variable(tf.truncated_normal([out_len], stddev=0.1), name="bl")
    yl = tf.matmul(xl_drop, Wl) + bl

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
        if i % 1000 == 0:
            train_x, train_y = trajactory.train.next_batch(minibatch)
            train_loss = predict_loss.eval(feed_dict={x1: train_x, y_: train_y, keep_prob: 1.0})
            test_x, test_y = trajactory.test.next_batch(minibatch)
            test_loss = predict_loss.eval(feed_dict={x1: test_x, y_: test_y, keep_prob: 1.0})
            print("step %d, t_loss %g, v_loss %g" % (i, train_loss, test_loss))

        # variables storing
        if (i % 50000 == 0) and (i != 0):
            saver.save(sess, savedModel)
            print("model saved.")

        # training batch
        batch_xs, batch_ys = trajactory.train.next_batch(minibatch)
        train_step.run(feed_dict={x1: batch_xs, y_: batch_ys, keep_prob: dropout, learn_rate: learnrate})

    saver.save(sess, savedModel)
    print("model saved.")


if __name__ == '__main__':
    tf.app.run()

