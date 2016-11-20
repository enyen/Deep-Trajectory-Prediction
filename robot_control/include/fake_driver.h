#ifndef FAKEDRIVER_H
#define FAKEDRIVER_H

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TransformStamped.h>
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
    tf::TransformBroadcaster m_tf_boardcaster;
    geometry_msgs::TransformStamped m_odom_trans;
    geometry_msgs::Twist m_twist_cmd;

    void handle_twist(const geometry_msgs::TwistConstPtr& msg);
};

#endif // FAKEDRIVER_H
