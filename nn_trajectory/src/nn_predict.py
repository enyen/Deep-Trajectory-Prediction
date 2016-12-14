#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rospy
from nav_msgs.msg import *
from geometry_msgs.msg import *
from std_msgs.msg import *
from human_motions.srv import *

import numpy as np
import tensorflow as tflow
import math
from collections import deque

traj_back = 30
traj_front = 30
order = 2


class nnPredict:
    path_pass = []

    def __init__(self):
        model_file = rospy.get_param('~model_file', '')
        model_meta_file = rospy.get_param('~model_meta_file', '')
        stat_file = rospy.get_param('~stat_file', '')
        if model_file == '' or model_meta_file == '' or stat_file == '':
            rospy.logerr('Incomplete files provided, check launch file.')
            sys.exit()

        self.predict_step = int(math.ceil(rospy.get_param('~predict_time', 2) * 10))

        self.sess = tflow.InteractiveSession()
        saver = tflow.train.import_meta_graph(model_meta_file)
        saver.restore(self.sess, model_file)
        self.predict = tflow.get_collection('yl')[0]
        self.input = tflow.get_collection('x1')[0]
        self.drop = tflow.get_collection('keep_prob')[0]
        self.stat = np.load(stat_file)

        self.learning = rospy.get_param('~online_learning', 0)
        self.train_step = tflow.get_collection('train_online')[0]
        self.y_ = tflow.get_collection('y_')[0]
        self.learn_rate = tflow.get_collection('learn_rate')[0]
        self.momentum = tflow.get_collection('momentum')[0]
        self.train_data = deque(maxlen=traj_front)
        self.train_label = deque(maxlen=traj_front)

        rospy.loginfo("NeuralNet Trajectory Prediction started.")
        if self.learning:
            rospy.loginfo("NeuralNet Online Learning Enabled.")
        else:
            rospy.loginfo("NeuralNet Online Learning Disabled.")

        self.sub_path_pass = rospy.Subscriber('/human_traj/path_pass', Path, self.handle_path)
        self.pub_path_param = rospy.Publisher('/human_traj/nn_param', Float64MultiArray, queue_size=1)
        self.pub_path_predict = rospy.Publisher('/human_traj/path_nn', Path, queue_size=1)
        self.server = rospy.Service('~pose2params_nn', path2params, self.handle_path2params)

    def handle_path(self, msg):
        batch_xs, params = self.predict_once(msg)
        # batch_xs = self.predict_recursive(msg, start_idx)
        self.pub_path_param.publish(params)

        ### online learning ###
        if self.learning:
            self.online_learning(batch_xs)

    def predict_once(self, msg, pub=True):
        start_idx = 0
        if len(msg.poses) < traj_back:
            rospy.logerr('insufficient pass path length.')
            return
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
        # batch_xs -= self.stat[0]
        # batch_xs /= self.stat[1]
        batch_xs = np.reshape(batch_xs, newshape=[1, traj_back * 3])
        result = self.predict.eval(feed_dict={self.input: batch_xs, self.drop: 1.0}, session=self.sess)

        ### Output Organisation ###
        new_poses = []
        # result = np.reshape(result, [traj_front, 2])
        # result *= self.stat[1, 1:]
        # result += self.stat[0, 1:]
        # lasttime = msg.poses[-1].header.stamp
        # X = np.zeros([traj_front, order + 1])
        # Y = np.zeros([traj_front, 2])
        # for j in range(0, traj_front):
        #     new_pose = PoseStamped()
        #     new_pose.pose.position.x = result[j, 0]
        #     new_pose.pose.position.y = result[j, 1]
        #     new_pose.pose.position.z = 0
        #     new_pose.pose.orientation.w = 1
        #     new_pose.header = msg.poses[-1].header
        #     # new_pose.header.stamp = lasttime + rospy.Duration(result[j, 0])
        #     # lasttime = new_pose.header.stamp
        #     new_poses.append(new_pose)
        #
        #     # for k in range(0, order + 1):
        #     #     X[j, k] = np.power(result[j, 0], k)
        #     # Y[i, 0] = result[j, 1]
        #     # Y[i, 1] = result[j, 2]

        # param = np.reshape(np.linalg.solve(X.transpose().dot(X), X.transpose().dot(Y)).transpose(), -1)
        result = np.reshape(result, -1)
        params = []
        for dt in range(0, 6):
            # params.append(param[dt])
            params.append(result[dt])
        params.append(msg.poses[0].header.stamp.to_sec())
        params.append(msg.poses[-1].header.stamp.to_sec())

        # span = (lasttime-msg.poses[0].header.stamp).to_sec() - 0.1
        # span = 0
        # for j in range(0, self.predict_step):
        #     # span += ((0.01-self.stat[0,0]) / self.stat[1,0])
        #     span += 0.1
        #     new_pose = PoseStamped()
        #     new_pose.pose.position.x = (
        #     result[0] + result[1] * span + result[2] * np.power(span, 2))  # *self.stat[1,1] + self.stat[0,1]
        #     new_pose.pose.position.y = (
        #     result[3] + result[4] * span + result[5] * np.power(span, 2))  # *self.stat[1,2] + self.stat[0,2]
        #     new_pose.pose.position.z = 0
        #     new_pose.pose.orientation.w = 1
        #     new_pose.header = msg.poses[-1].header
        #     # new_pose.header.stamp = msg.poses[0].header.stamp + rospy.Duration(span)
        #     new_poses.append(new_pose)
        for j in range(0, self.predict_step):
            if (len(result)/2) < j:
                break
            new_pose = PoseStamped()
            new_pose.pose.position.x = msg.poses[-1].pose.position.x + result[j*2]
            new_pose.pose.position.y = msg.poses[-1].pose.position.y + result[j*2+1]
            new_pose.pose.position.z = 0
            new_pose.pose.orientation.w = 1
            new_pose.header = msg.poses[-1].header
            new_poses.append(new_pose)

        if pub:
            self.pub_path_predict.publish(header=msg.header, poses=new_poses)

        return batch_xs, Float64MultiArray(data=params)

    def predict_recursive(self, msg, start_idx):
        batch_xs = np.zeros([traj_back+self.predict_step, 3])
        lasttime = msg.poses[start_idx].header.stamp
        for i in range(start_idx, len(msg.poses)):
            batch_xs[i, 0] = (msg.poses[i].header.stamp - lasttime).to_sec()
            batch_xs[i, 1] = msg.poses[i].pose.position.x
            batch_xs[i, 2] = msg.poses[i].pose.position.y
            lasttime = msg.poses[i].header.stamp

        new_poses = []
        smallstep = 2
        for k in range(0, self.predict_step, smallstep):
            batch_xs_ = np.reshape(batch_xs[k:len(msg.poses)+k, :], newshape=[1, traj_back * 3])
            result = self.predict.eval(feed_dict={self.input: batch_xs_, self.drop: 1.0}, session=self.sess)
            result = np.reshape(result, -1)

            span = 0
            for j in range(0, smallstep):
                span += 0.1
                new_pose = PoseStamped()
                new_pose.header = msg.poses[-1].header
                new_pose.header.stamp = msg.poses[-1].header.stamp + rospy.Duration(span)
                new_pose.pose.position.x = (result[0] + result[1] * span + result[2] * np.power(span, 2))
                new_pose.pose.position.y = (result[3] + result[4] * span + result[5] * np.power(span, 2))
                new_pose.pose.position.z = 0
                new_poses.append(new_pose)

                batch_xs[len(msg.poses)+j+k, 0] = 0.1
                batch_xs[len(msg.poses)+j+k, 1] = new_pose.pose.position.x
                batch_xs[len(msg.poses)+j+k, 2] = new_pose.pose.position.y

        self.pub_path_predict.publish(header=msg.header, poses=new_poses)

        # params = []
        # for dt in range(0, 6):
        #     params.append(result[dt])
        # params.append(msg.poses[0].header.stamp.to_sec())
        # params.append(msg.poses[-1].header.stamp.to_sec())
        # self.pub_path_param.publish(Float64MultiArray(data=params))

        return np.reshape(batch_xs[:len(msg.poses), :], newshape=[1, traj_back * 3])

    def online_learning(self, batch_xs):
        if len(self.train_label) > 0:
            distance = (batch_xs[0, -2] - self.train_label[-1][1]) ** 2 + \
                       (batch_xs[0, -1] - self.train_label[-1][2]) ** 2
            if np.sqrt(distance) > 1.5:  # 1.88
                rospy.logwarn("trajectory step too big, recollecting training data...")
                self.train_data.clear()
                self.train_label.clear()

        self.train_label.append(batch_xs[0, -3:])
        if len(self.train_label) == traj_front:
            # X = np.zeros([traj_front, order + 1])
            # Y = np.zeros([traj_front, 2])
            # accum = 0
            # for i in range(0, traj_front):
            #     accum += self.train_label[i][0]
            #     for j in range(0, order + 1):
            #         X[i, j] = np.power(accum, j)
            #     Y[i, 0] = self.train_label[i][1]
            #     Y[i, 1] = self.train_label[i][2]
            X = np.zeros([traj_back+traj_front, order + 1])
            Y = np.zeros([traj_back+traj_front, 2])
            accum = 0
            for i in range(0, traj_back+traj_front):
                if i < traj_back:
                    accum += self.train_data[0][0, i*3]
                    for j in range(0, order + 1):
                        X[i, j] = np.power(accum, j)
                    Y[i, 0] = self.train_data[0][0, i*3+1]
                    Y[i, 1] = self.train_data[0][0, i*3+2]
                else:
                    accum += self.train_label[i-traj_back][0]
                    for j in range(0, order + 1):
                        X[i, j] = np.power(accum, j)
                    Y[i, 0] = self.train_label[i-traj_back][1]
                    Y[i, 1] = self.train_label[i-traj_back][2]

            temp = np.linalg.solve(X.transpose().dot(X), X.transpose())
            self.train_step.run(feed_dict={self.input: self.train_data[0],
                                           self.y_: np.reshape(temp.dot(Y).transpose(), [1, 6]),
                                           self.learn_rate: 5 * 1e-4,
                                           self.momentum: 0.7,
                                           self.drop: 0.7}, session=self.sess)
        self.train_data.append(batch_xs)

    def handle_path2params(self, req):
        _, param = self.predict_once(req.path, pub=False)
        return path2paramsResponse(param)


def main():
    rospy.init_node('nn_predict', anonymous=True)
    nnPredict()
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

