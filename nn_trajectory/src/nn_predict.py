#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import rospy
from nav_msgs.msg import _Odometry
from nav_msgs.msg import _Path

import numpy as np
import tensorflow as tf


class nn_predict:
    def __init__(self):
        self.pub_path_pass = rospy.Publisher('/human_pose/path_pass_nn', _Path.Path, queue_size=1)
        self.pub_path_predict = rospy.Publisher('/human_pose/path_future_nn', _Path.Path, queue_size=1)
        odom_topic = rospy.get_param('~odom_topic', '')
        if odom_topic == '':
            rospy.logerr('No odom_topic provided, check launch file.')
            sys.exit()
        rospy.Subscriber('/bill', _Odometry.Odometry, self.handle_odom)

        model_file = rospy.get_param('~model_file', '')
        model_meta_file = rospy.get_param('~model_meta_file', '')
        if model_file == '' or model_meta_file == '':
            rospy.logerr('No model_file provided, check launch file.')
            sys.exit()

        self.sess = tf.InteractiveSession()
        saver = tf.train.import_meta_graph(model_meta_file)
        saver.restore(self.sess, model_file)
        self.predict = tf.get_collection('y3')[0]

    def handle_odom(self, data):
        # print (self.predict.eval(feed_dict={x1: train_x, keep_prob: 1.0}))
        print(self.predict.eval(feed_dict={x1: batch_xs, keep_prob: 0.5}, session=self.sess))


def main():
    rospy.init_node('nn_predict', anonymous=True)
    nn_predict()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

