#ifndef FAKEDRIVER_H
#define FAKEDRIVER_H

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <tf/transform_broadcaster.h>

class FakeDriver
{
public:
    FakeDriver();
    ~FakeDriver();

    void step();
    double control_freq;

private:
    ros::NodeHandle m_nh;
    ros::Subscriber m_sub_twist;
    ros::Subscriber m_sub_grid;
    tf::TransformBroadcaster m_tf_boardcaster;
    geometry_msgs::TransformStamped m_odom_trans;
    geometry_msgs::Twist m_twist_cmd;
    nav_msgs::OccupancyGrid m_grid;

    void handle_twist(const geometry_msgs::TwistConstPtr& msg);
    void handle_grid(const nav_msgs::OccupancyGridConstPtr &msg);
};

#endif // FAKEDRIVER_H
