#ifndef PASSPATH_H
#define PASSPATH_H

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <vector>

class PassPath
{
public:
    PassPath();
    ~PassPath();

private:
    ros::NodeHandle m_nh;
    ros::Subscriber m_sub_odom;
    ros::Publisher m_pub_path;

    void handle_odom(const nav_msgs::OdometryConstPtr& msg);

    unsigned int m_path_size;
    std::vector<geometry_msgs::PoseStamped> m_path_pose;
};

#endif // PASSPATH_H
