#include "fake_driver.h"
#include <math.h>

FakeDriver::FakeDriver()
    : m_nh("~")
{
    control_freq = m_nh.param("controller_freq", 15.0);
    ROS_INFO_STREAM("robot controller running at " << control_freq << "hz");

    std::string vel_topic = m_nh.param("vel_topic", std::string(""));
    if(vel_topic.empty())
    {
        ROS_ERROR_STREAM("no cmd_vel provided. Check launch file.");
        std::abort();
    }
//    m_sub_twist = m_nh.subscribe(vel_topic, 1, &FakeDriver::handle_twist, this);
    ros::NodeHandle nh;
    m_sub_twist = nh.subscribe("cmd_vel", 1, &FakeDriver::handle_twist, this);

    m_odom_trans.header.frame_id = "map";
    m_odom_trans.child_frame_id = "bot";
    m_odom_trans.header.stamp = ros::Time::now();
    m_odom_trans.transform.translation.x = 2.5;
    m_odom_trans.transform.translation.y = 1.5;
    m_odom_trans.transform.translation.z = 0;
    m_odom_trans.transform.rotation = tf::createQuaternionMsgFromYaw(0);
    m_tf_boardcaster.sendTransform(m_odom_trans);
}

FakeDriver::~FakeDriver()
{

}

void FakeDriver::handle_twist(const geometry_msgs::TwistConstPtr &msg)
{
    m_twist_cmd = *msg;
}

void FakeDriver::step()
{
    m_odom_trans.header.stamp = ros::Time::now();
    double timelapsed = 1.0/control_freq;

    tf::Quaternion q;
    q.setZ(m_odom_trans.transform.rotation.z);
    q.setW(m_odom_trans.transform.rotation.w);
    double heading = q.getAngle() * (m_odom_trans.transform.rotation.z<0 ? -1 : 1);

    m_odom_trans.transform.translation.x += m_twist_cmd.linear.x*cos(heading)*timelapsed -
                                            m_twist_cmd.linear.y*sin(heading)*timelapsed;
    m_odom_trans.transform.translation.y += m_twist_cmd.linear.x*sin(heading)*timelapsed +
                                            m_twist_cmd.linear.y*cos(heading)*timelapsed;
    m_odom_trans.transform.translation.z = 0;
    heading = heading + m_twist_cmd.angular.z*timelapsed;
    heading = fmod(fabs(heading), 2*M_PI) * ((heading<0) ? -1 : 1);

    m_odom_trans.transform.rotation = tf::createQuaternionMsgFromYaw(heading);

    m_tf_boardcaster.sendTransform(m_odom_trans);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "fake_driver");
    FakeDriver fake_driver;
    ros::Rate controll_loop(fake_driver.control_freq);

    while(ros::ok())
    {
        ros::spinOnce();
        fake_driver.step();
        controll_loop.sleep();
    }

    return 0;
}
