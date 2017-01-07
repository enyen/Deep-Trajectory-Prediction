#ifndef POSE_2_RAY_H
#define POSE_2_RAY_H

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <eigen3/Eigen/Dense>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include <vector>

#include <sstream>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

class pose_2_ray
{
    typedef Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic> MatrixXc;

public:
    pose_2_ray();
    ~pose_2_ray();

private:
    void handle_map(const nav_msgs::OccupancyGridConstPtr &msg);
    void handle_odom(const nav_msgs::OdometryConstPtr &msg);

    ros::NodeHandle m_nh;
    ros::Subscriber m_sub_map;
    ros::Subscriber m_sub_odom;
    ros::Publisher m_pub_scan;

    sensor_msgs::LaserScan m_scan;
    geometry_msgs::TransformStamped m_pose;
    tf2_ros::TransformBroadcaster tfb;

    nav_msgs::MapMetaData m_mapInfo;
    MatrixXc m_mapData;
    tf::StampedTransform m_mapTransform;
    bool m_map_received;

    std::ofstream m_myfile;
    double m_time_start;
};

#endif // POSE_2_RAY_H
