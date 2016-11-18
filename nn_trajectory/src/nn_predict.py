#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rospy
from nav_msgs.msg import *
from geometry_msgs.msg import *
from std_msgs.msg import *
import tf.transformations

import numpy as np
import tensorflow as tflow

traj_back = 30
traj_front = 30
order = 2


class nnPredict:
    path_pass = []

    def __init__(self):
        self.sub_path_pass = rospy.Subscriber('/human_traj/path_pass', Path, self.handle_path)
        self.pub_path_param = rospy.Publisher('/human_traj/nn_param', Float64MultiArray, queue_size=1)
        self.pub_path_predict = rospy.Publisher('/human_traj/path_nn', Path, queue_size=1)

        model_file = rospy.get_param('~model_file', '')
        model_meta_file = rospy.get_param('~model_meta_file', '')
        stat_file = rospy.get_param('~stat_file', '')
        if model_file == '' or model_meta_file == '' or stat_file == '':
            rospy.logerr('Incomplete files provided, check launch file.')
            sys.exit()

        self.predict_time = rospy.get_param('~predict_time', 2)

        self.sess = tflow.InteractiveSession()
        saver = tflow.train.import_meta_graph(model_meta_file)
        saver.restore(self.sess, model_file)
        self.predict = tflow.get_collection('yl')[0]
        self.input = tflow.get_collection('x1')[0]
        self.drop = tflow.get_collection('keep_prob')[0]
        self.stat = np.load(stat_file)

    def handle_path(self, msg):
        start_idx = 0
        if len(msg.poses) < traj_back:
            rospy.logerr('insufficient pass path length.')
            sys.exit()
        else:
            start_idx = len(msg.poses) - traj_back

        #### Data Preparation ###
        batch_xs = np.zeros([traj_back, 3])
        lasttime = msg.poses[start_idx].header.stamp
        for i in range(start_idx, len(msg.poses)):
            # batch_xs[i, 0] = (msg.poses[i].header.stamp - msg.poses[0].header.stamp).to_sec()
            batch_xs[i, 0] = (msg.poses[i].header.stamp - lasttime).to_sec()
            batch_xs[i, 1] = msg.poses[i].pose.position.x
            batch_xs[i, 2] = msg.poses[i].pose.position.y
            lasttime = msg.poses[i].header.stamp
        batch_xs -= self.stat[0]
        batch_xs /= self.stat[1]
        batch_xs = np.reshape(batch_xs, newshape=[1,traj_back*3])
        result = self.predict.eval(feed_dict={self.input: batch_xs, self.drop: 1.0}, session=self.sess)

        ### Output Organisation ###
        new_poses = []
        result = np.reshape(result, [traj_front, 3])
        result *= self.stat[1]
        result += self.stat[0]
        lasttime = msg.poses[-1].header.stamp
        X = np.zeros([traj_front, order + 1])
        Y = np.zeros([traj_front, 2])
        for j in range(0, traj_front):
            new_pose = PoseStamped()
            new_pose.pose.position.x = result[j, 1]
            new_pose.pose.position.y = result[j, 2]
            new_pose.pose.position.z = 0
            new_pose.pose.orientation.w = 1
            new_pose.header = msg.poses[-1].header
            new_pose.header.stamp = lasttime + rospy.Duration(result[j, 0])
            lasttime = new_pose.header.stamp
            new_poses.append(new_pose)

            for k in range(0, order + 1):
                X[j, k] = np.power(result[j, 0], k)
            Y[i, 0] = result[j, 1]
            Y[i, 1] = result[j, 2]

        param = np.reshape(np.linalg.solve(X.transpose().dot(X), X.transpose().dot(Y)).transpose(), -1)
        # result = np.reshape(np.reshape(result, [order + 1, 2]).transpose(), -1)
        params = []
        for dt in range(0, 6):
            params.append(param[dt])
            # params.append(result[dt])
        params.append(msg.poses[0].header.stamp.to_sec())
        params.append(msg.poses[-1].header.stamp.to_sec())
        self.pub_path_param.publish(Float64MultiArray(data=params))

        # span = (msg.poses[-1].header.stamp - msg.poses[0].header.stamp).to_sec()
        # for j in range(0, self.predict_time*10):
        #     span += 0.1
        #     new_pose = PoseStamped()
        #     new_pose.pose.position.x = result[0] + result[1] * span + result[2] * np.power(span, 2)
        #     new_pose.pose.position.y = result[3] + result[4] * span + result[5] * np.power(span, 2)
        #     new_pose.pose.position.z = 0
        #     new_pose.pose.orientation = msg.poses[-1].pose.orientation
        #     new_pose.header = msg.poses[-1].header
        #     new_pose.header.stamp = msg.poses[0].header.stamp + rospy.Duration(span)
        #     new_poses.append(new_pose)

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

