#ifndef POSE_2_RAY_H
#define POSE_2_RAY_H

#include <ros/ros.h>
#include <map_ray_caster/map_ray_caster.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <geometry_msgs/Transform.h>

#include <sstream>
#include <vector>

class pose_2_ray
{
    typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXi_rowMajor;

public:
    pose_2_ray();
    ~pose_2_ray();

private:
    void handleMap(const nav_msgs::OccupancyGridConstPtr &msg);
    void handleOdom(const nav_msgs::OdometryConstPtr &msg);
    void handleMocap(const geometry_msgs::TransformConstPtr &msg);

    ros::NodeHandle m_nh;
    ros::Subscriber m_sub_map;
    ros::Subscriber m_sub_odom;
    ros::Subscriber m_sub_mocap;
    ros::Publisher m_pub_scan;

    ros::Publisher asd;
    float range;

    std::string m_occupancy_map_topic;
    nav_msgs::OccupancyGrid m_map;
    sensor_msgs::LaserScan m_scan;
    geometry_msgs::Transform m_map_transform;
};

#endif // POSE_2_RAY_H
