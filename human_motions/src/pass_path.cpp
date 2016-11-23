#include "human_motions/pass_path.h"

PassPath::PassPath()
    : m_nh("~")
{
    m_path_size = m_nh.param("path_size", 30);
    std::string odom_topic = m_nh.param("odom_topic", std::string(""));
    if(odom_topic.empty())
    {
        ROS_INFO_STREAM("no odom_topic provided. Check launch file.");
        std::abort();
    }
    m_sub_odom = m_nh.subscribe(odom_topic, 1, &PassPath::handle_odom, this);
    m_pub_path = m_nh.advertise<nav_msgs::Path>("/human_traj/path_pass", 1);

    ROS_INFO_STREAM("Node collecting and publishing passed-trajectory started.");
}

PassPath::~PassPath()
{

}

void PassPath::handle_odom(const nav_msgs::OdometryConstPtr &msg)
{
    if(m_path_pose.size() != 0)
    {
        double dist = pow(m_path_pose[m_path_pose.size()-1].pose.position.x - msg->pose.pose.position.x, 2) +
                      pow(m_path_pose[m_path_pose.size()-1].pose.position.y - msg->pose.pose.position.y, 2);
        if(sqrt(dist) > 2.0) // 2.141
        {
            ROS_WARN_STREAM("trajectory step too big, recollecting trajectory...");
            m_path_pose.clear();
        }
    }

    if(m_path_pose.size() >= m_path_size)
    {
        m_path_pose.erase(m_path_pose.begin());
    }

    geometry_msgs::PoseStamped new_pose;
    new_pose.header = msg->header;
    new_pose.pose = msg->pose.pose;
    new_pose.pose.position.z = 0;
    new_pose.pose.orientation.w = 1;
    m_path_pose.push_back(new_pose);

    if(m_path_pose.size() == m_path_size)
    {
        nav_msgs::Path nav_path;
        nav_path.header = msg->header;
        nav_path.poses = std::vector<geometry_msgs::PoseStamped>(m_path_pose.begin(), m_path_pose.begin()+m_path_size);
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
