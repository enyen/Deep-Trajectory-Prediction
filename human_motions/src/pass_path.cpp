#include "human_motions/pass_path.h"

PassPath::PassPath()
    : m_nh("~")
{
    m_path_size = m_nh.param("path_size", 20);
    std::string odom_topic = m_nh.param("odom_topic", std::string(""));
    if(odom_topic.empty())
    {
        ROS_INFO_STREAM("no odom_topic provided. Check launch file.");
        std::abort();
    }
    m_sub_odom = m_nh.subscribe(odom_topic, 1, &PassPath::handle_odom, this);
    m_pub_path = m_nh.advertise<nav_msgs::Path>("/human_traj/path_pass", 1);
}

PassPath::~PassPath()
{

}

void PassPath::handle_odom(const nav_msgs::OdometryConstPtr &msg)
{
    if(m_path_pose.size() >= m_path_size)
        m_path_pose.erase(m_path_pose.begin());

    geometry_msgs::PoseStamped new_pose;
    new_pose.header = msg->header;
    new_pose.pose = msg->pose.pose;
    new_pose.pose.position.z = 0;
    m_path_pose.push_back(new_pose);

    if(m_path_pose.size() == m_path_size)
    {
        nav_msgs::Path nav_path;
        nav_path.header = msg->header;
        nav_path.poses = m_path_pose;
        m_pub_path.publish(nav_path);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pass_path");
    PassPath pp;
    ros::spin();

    return 0;
}
