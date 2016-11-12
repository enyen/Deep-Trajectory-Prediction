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

traj_back = 30
traj_front = 50


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
        self.predict = tf.get_collection('yl')[0]
        self.input = tf.get_collection('x1')[0]
        self.drop = tf.get_collection('keep_prob')[0]

    def handle_odom(self, msg):
        if len(self.path_pass) >= traj_back:
            self.path_pass.pop(0)

        self.path_pass.append(PoseStamped(header=msg.header, pose=msg.pose.pose))

        if len(self.path_pass) == traj_back:
            self.pub_path_pass.publish(Path(header=msg.header, poses=self.path_pass))

            batch_xs = np.zeros([traj_back, 3])
            lasttime = self.path_pass[0].header.stamp
            for i in range(0, traj_back):
                # batch_xs[i, 0] = (self.path_pass[i].header.stamp - self.path_pass[0].header.stamp).to_sec()
                batch_xs[i, 0] = (self.path_pass[i].header.stamp - lasttime).to_sec()
                batch_xs[i, 1] = self.path_pass[i].pose.position.x
                batch_xs[i, 2] = self.path_pass[i].pose.position.y
                lasttime = self.path_pass[i].header.stamp
            result = self.predict.eval(feed_dict={self.input: np.reshape(batch_xs, newshape=[1,traj_back*3]), self.drop: 1.0}, session=self.sess)
            # result = np.reshape(result, [traj_front, 3])
            result = np.reshape(result, [traj_front, 2])
            new_poses = []
            for j in range(0, traj_front):
                new_pose = Pose()
                new_pose.position.x = result[j, 0]
                new_pose.position.y = result[j, 1]
                new_pose.position.z = msg.pose.pose.position.z
                new_pose.orientation = msg.pose.pose.orientation
                new_poses.append(PoseStamped(header=msg.header, pose=new_pose))
            # new_poses = []
            # span = (self.path_pass[traj_back-1].header.stamp - self.path_pass[0].header.stamp).to_sec()
            # result /= 20
            # for j in range(0, traj_front):
            #     span += 0.051
            #     new_pose = Pose()
            #     new_pose.position.x = result[0,0] + result[0,1] * span + result[0,2] * np.power(span, 2)
            #     new_pose.position.y = result[0,3] + result[0,4] * span + result[0,4] * np.power(span, 2)
            #     new_pose.position.z = msg.pose.pose.position.z
            #     new_pose.orientation = msg.pose.pose.orientation
            #     new_poses.append(PoseStamped(header=msg.header, pose=new_pose))

            self.pub_path_predict.publish(header=msg.header, poses=new_poses)



def main():
    rospy.init_node('nn_predict', anonymous=True)
    nnPredict()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

