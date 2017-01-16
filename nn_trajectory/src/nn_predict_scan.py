#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rospy
from nav_msgs.msg import *
from geometry_msgs.msg import *
from std_msgs.msg import *
from tf import *

import numpy as np
import tensorflow as tflow
import math
from collections import deque

traj_back = 30
traj_front = 30
scan_size = 173


class nnPredict:

    def __init__(self):
        model_file = rospy.get_param('~model_file', '')
        model_meta_file = rospy.get_param('~model_meta_file', '')
        if model_file == '' or model_meta_file == '':
            rospy.logerr('Incomplete files provided, check launch file.')
            sys.exit()

        self.predict_step = int(math.ceil(rospy.get_param('~predict_time', 2) * 10))

        self.sess = tflow.InteractiveSession()
        saver = tflow.train.import_meta_graph(model_meta_file)
        saver.restore(self.sess, model_file)
        self.predict = tflow.get_collection('yl')[0]
        self.input = tflow.get_collection('x1')[0]
        self.drop = tflow.get_collection('keep_prob')[0]
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

        occu_grid = rospy.get_param('~occupancy_grid', '/map')
        self.sub_map = rospy.Subscriber(occu_grid, OccupancyGrid, self.handle_map)
        self.sub_path_pass = rospy.Subscriber('/human_traj/path_pass', Path, self.handle_path)
        laser_scan = rospy.get_param('~laser_scans', '/human_traj/scan_pass')
        self.sub_scan = rospy.Subscriber(laser_scan, Float32MultiArray, self.handle_scans)
        self.pub_path_predict = rospy.Publisher('/human_traj/path_nn', Path, queue_size=1)

        self.pub_score = rospy.Publisher('/human_traj/nn_score', Float64MultiArray, queue_size=1)
        self.pub_score_map = rospy.Publisher('/human_traj/nn_score_map', OccupancyGrid, queue_size=1)

        self.map_info = None
        self.map_trans = None
        self.performance_map = None
        self.map_received = False
        self.score = 0
        self.posX = 0
        self.posY = 0
        self.pathHeader = []

    def handle_map(self, msg):
        # if self.map_received:
        #     return
        self.map_info = msg.info
        self.performance_map = np.zeros([self.map_info.width, self.map_info.height], dtype=np.int8)
        tflis = TransformListener()
        tflis.waitForTransform("/map", "/mocap", rospy.Time(0), rospy.Duration(5))
        trans, rot = tflis.lookupTransform("/map", "/mocap", rospy.Time(0))
        trans = Vector3(x=trans[0]/self.map_info.resolution, y=trans[1]/self.map_info.resolution, z=0)
        self.map_trans = Transform(translation=trans, rotation=rot)
        self.map_received = True
        rospy.loginfo('Map received.')

    def handle_path(self, msg):
        if not self.map_received:
            return
        self.posX = msg.poses[-1].pose.position.x
        self.posY = msg.poses[-1].pose.position.y
        self.pathHeader = msg.header

        # Performance measure
        predict_score = 0
        for i in range(0, len(msg.poses)):
            x = msg.poses[i].pose.position.x / self.map_info.resolution + self.map_trans.translation.x
            y = msg.poses[i].pose.position.y / self.map_info.resolution + self.map_trans.translation.y
            predict_score += self.performance_map[int(x), int(y)]
        smoother = 0.99
        self.score = smoother*self.score + (1-smoother)*predict_score
        self.pub_score.publish(Float64MultiArray(data=[self.score]))

    def handle_scans(self, msg):
        if not self.map_received:
            return

        # new prediction
        batch_xs, params = self.predict_once3(data=msg.data)
        # self.pub_path_param.publish(params)

        ### online learning ###
        # if self.learning:
        #     self.online_learning(batch_xs)

    def predict_once3(self, data, pub=True):
        start_idx = 0

        #### Data Preparation ###
        batch_xs = np.reshape(data, [1, 1, scan_size, traj_back], order='F')
        result = self.predict.eval(feed_dict={self.input: batch_xs, self.drop: 1.0}, session=self.sess)

        ### Output Organisation ###
        new_poses = []
        params = []
        result = np.reshape(result, -1)

        for j in range(0, self.predict_step):
            if (len(result)/2) < j:
                break
            new_pose = PoseStamped()
            new_pose.pose.position.x = self.posX + result[j*2]
            new_pose.pose.position.y = self.posY + result[j*2+1]
            new_pose.pose.position.z = 0
            new_pose.pose.orientation.w = 1
            new_pose.header = self.pathHeader
            new_poses.append(new_pose)

        if pub:
            self.pub_path_predict.publish(header=self.pathHeader, poses=new_poses)
            self.update_map(new_poses)

        return batch_xs, Float64MultiArray(data=params)

    def update_map(self, path):
        confidence = 0
        for i in range(0, len(path)):
            x = int(path[i].pose.position.x / self.map_info.resolution + self.map_trans.translation.x)
            y = int(path[i].pose.position.y / self.map_info.resolution + self.map_trans.translation.y)
            if x + 1 < self.map_info.width and x-1 >= 0 and y+1<self.map_info.height and y-1>=0:
                confidence = min(confidence + 3, 100)
                self.performance_map[x, y] = min(self.performance_map[x, y] + confidence, 100)
                self.performance_map[x+1, y] = min(self.performance_map[x+1, y] + confidence/3, 100)
                self.performance_map[x-1, y] = min(self.performance_map[x-1, y] + confidence/3, 100)
                self.performance_map[x, y+1] = min(self.performance_map[x, y+1] + confidence/3, 100)
                self.performance_map[x, y-1] = min(self.performance_map[x, y-1] + confidence/3, 100)
                self.performance_map[x+1, y+1] = min(self.performance_map[x+1, y+1] + confidence/4, 100)
                self.performance_map[x-1, y-1] = min(self.performance_map[x-1, y-1] + confidence/4, 100)
                self.performance_map[x+1, y-1] = min(self.performance_map[x+1, y-1] + confidence/4, 100)
                self.performance_map[x-1, y+1] = min(self.performance_map[x-1, y+1] + confidence/4, 100)

        self.performance_map = (self.performance_map.astype(np.double) / 1.1).astype(np.int8)
        self.pub_score_map.publish(OccupancyGrid(header=Header(frame_id='map', stamp=rospy.Time.now()),
                                                 info=self.map_info,
                                                 data=np.reshape(self.performance_map.transpose(), -1)))

    # def online_learning(self, batch_xs):
    #     if len(self.train_label) > 0:
    #         distance = (batch_xs[0, -2] - self.train_label[-1][1]) ** 2 + \
    #                    (batch_xs[0, -1] - self.train_label[-1][2]) ** 2
    #         if np.sqrt(distance) > 1.5:  # 1.88
    #             rospy.logwarn("trajectory step too big, recollecting training data...")
    #             self.train_data.clear()
    #             self.train_label.clear()
    #
    #     self.train_label.append(batch_xs[0, -3:])
    #     if len(self.train_label) == traj_front:
    #         # X = np.zeros([traj_front, order + 1])
    #         # Y = np.zeros([traj_front, 2])
    #         # accum = 0
    #         # for i in range(0, traj_front):
    #         #     accum += self.train_label[i][0]
    #         #     for j in range(0, order + 1):
    #         #         X[i, j] = np.power(accum, j)
    #         #     Y[i, 0] = self.train_label[i][1]
    #         #     Y[i, 1] = self.train_label[i][2]
    #         X = np.zeros([traj_back+traj_front, order + 1])
    #         Y = np.zeros([traj_back+traj_front, 2])
    #         accum = 0
    #         for i in range(0, traj_back+traj_front):
    #             if i < traj_back:
    #                 accum += self.train_data[0][0, i*3]
    #                 for j in range(0, order + 1):
    #                     X[i, j] = np.power(accum, j)
    #                 Y[i, 0] = self.train_data[0][0, i*3+1]
    #                 Y[i, 1] = self.train_data[0][0, i*3+2]
    #             else:
    #                 accum += self.train_label[i-traj_back][0]
    #                 for j in range(0, order + 1):
    #                     X[i, j] = np.power(accum, j)
    #                 Y[i, 0] = self.train_label[i-traj_back][1]
    #                 Y[i, 1] = self.train_label[i-traj_back][2]
    #
    #         temp = np.linalg.solve(X.transpose().dot(X), X.transpose())
    #         self.train_step.run(feed_dict={self.input: self.train_data[0],
    #                                        self.y_: np.reshape(temp.dot(Y).transpose(), [1, 6]),
    #                                        self.learn_rate: 5 * 1e-4,
    #                                        self.momentum: 0.7,
    #                                        self.drop: 0.7}, session=self.sess)
    #     self.train_data.append(batch_xs)


def main():
    rospy.init_node('nn_predict_scan', anonymous=True)
    nnPredict()
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
