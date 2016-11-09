#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import rospy
from nav_msgs.msg import *
from geometry_msgs.msg import *

import numpy as np
import tensorflow as tf

traj_back = 20


class nnPredict:
    path_pass = []

    def __init__(self):
        self.pub_path_pass = rospy.Publisher('/human_pose/path_pass_nn', Path, queue_size=1)
        self.pub_path_predict = rospy.Publisher('/human_pose/path_future_nn', Path, queue_size=1)
        odom_topic = rospy.get_param('~odom_topic', '')
        if odom_topic == '':
            rospy.logerr('No odom_topic provided, check launch file.')
            sys.exit()
        rospy.Subscriber('/bill', Odometry, self.handle_odom)

        model_file = rospy.get_param('~model_file', '')
        model_meta_file = rospy.get_param('~model_meta_file', '')
        if model_file == '' or model_meta_file == '':
            rospy.logerr('No model_file provided, check launch file.')
            sys.exit()

        self.sess = tf.InteractiveSession()
        saver = tf.train.import_meta_graph(model_meta_file)
        saver.restore(self.sess, model_file)
        self.predict = tf.get_collection('y3')[0]
        self.input = tf.get_collection('x1')[0]
        self.drop = tf.get_collection('keep_prob')[0]

    def handle_odom(self, msg):
        if len(self.path_pass) >= traj_back:
            self.path_pass.pop(0)

        self.path_pass.append(PoseStamped(header=msg.header, pose=msg.pose.pose))

        if len(self.path_pass) == traj_back:
            self.pub_path_pass.publish(Path(header=msg.header, poses=self.path_pass))
            batch_xs = np.zeros([traj_back, 3])
            for i in range(0, traj_back):
                batch_xs[i, 0] = (self.path_pass[i].header.stamp - self.path_pass[0].header.stamp).to_sec()
                batch_xs[i, 1] = self.path_pass[i].pose.position.x
                batch_xs[i, 2] = self.path_pass[i].pose.position.y
            batch_xs = batch_xs - np.mean(batch_xs, axis=0)
            batch_xs = batch_xs / np.std(batch_xs, axis=0)
            print(self.predict.eval(feed_dict={self.input: np.reshape(batch_xs, newshape=[1,traj_back*3]), self.drop: 0.5}, session=self.sess))


def main():
    rospy.init_node('nn_predict', anonymous=True)
    nnPredict()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

