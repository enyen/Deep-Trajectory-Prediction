#ifndef PASSPATH_H
#define PASSPATH_H

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Float32MultiArray.h>
#include <vector>
#include <math.h>

class PassPath
{
public:
    PassPath();
    ~PassPath();

private:
    ros::NodeHandle m_nh;
    ros::Subscriber m_sub_odom;
    ros::Publisher m_pub_path;
    ros::Subscriber m_sub_scan;
    ros::Publisher m_pub_scan_path;

    void handle_odom(const nav_msgs::OdometryConstPtr& msg);
    void handle_scan(const sensor_msgs::LaserScanConstPtr& msg);

    unsigned int m_path_size;
    unsigned int m_path_scan_size;
    std::vector<geometry_msgs::PoseStamped> m_path_pose;
    std::vector<float> m_path_scan;
};

#endif // PASSPATH_H
