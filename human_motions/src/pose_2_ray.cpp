#include "human_motions/pose_2_ray.h"

pose_2_ray::pose_2_ray()
    : m_nh("~")
    , m_map_received(false)
{
    std::string grid_topic = m_nh.param("occupancy_grid", std::string(""));
    if(grid_topic.empty())
    {
        ROS_ERROR_STREAM("no occupancy_grid provided. Check launch file.");
        std::abort();
    }
    m_sub_map = m_nh.subscribe(grid_topic, 1, &pose_2_ray::handle_map, this);

    std::string odom_topic = m_nh.param("odom_topic", std::string(""));
    if(odom_topic.empty())
    {
        ROS_INFO_STREAM("no odom_topic provided. Check launch file.");
        std::abort();
    }
    m_sub_odom = m_nh.subscribe(odom_topic, 1, &pose_2_ray::handle_odom, this);

    m_pub_scan = m_nh.advertise<sensor_msgs::LaserScan>("/pose_2_ray/scan", 1);

    m_scan.angle_min = -2;
    m_scan.angle_max = 2;
    m_scan.angle_increment = 2.0*M_PI/180.0;
    m_scan.ranges.resize(((m_scan.angle_max-m_scan.angle_min)/m_scan.angle_increment) + 1);
    m_scan.range_max = 3.0;
    m_scan.range_min = 0;
    m_scan.header.frame_id = "cpose";

    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "/mocap";
    pose.child_frame_id = "/cpose";
    pose.transform.translation.x = 0;
    pose.transform.translation.y = 0;
    pose.transform.translation.z = 0;
    pose.transform.rotation.w = 1;
    tfb.sendTransform(pose);
}

pose_2_ray::~pose_2_ray()
{

}

void pose_2_ray::handle_map(const nav_msgs::OccupancyGridConstPtr &msg)
{
    m_mapInfo = msg->info;
    std::vector<signed char> grid = std::vector<signed char>(msg->data.begin(), msg->data.end());
    m_mapData = Eigen::Map<MatrixXc>(grid.data(), m_mapInfo.width, m_mapInfo.height);

    tf::TransformListener listener;
    listener.waitForTransform("/map", "/mocap", msg->header.stamp, ros::Duration(5));
    listener.lookupTransform("/map", "/mocap", ros::Time(0), m_mapTransform);
    ROS_INFO_STREAM("[" <<  m_mapTransform.getOrigin().x() << " " <<  m_mapTransform.getOrigin().y()<< "] ");
    m_mapTransform.setOrigin(tf::Vector3(m_mapTransform.getOrigin().x()/m_mapInfo.resolution,
                                         m_mapTransform.getOrigin().y()/m_mapInfo.resolution,
                                         0));
    ROS_INFO_STREAM("[" <<  m_mapTransform.getOrigin().x() << " " <<  m_mapTransform.getOrigin().y()<< "] ");
    m_map_received = true;
}

void pose_2_ray::handle_odom(const nav_msgs::OdometryConstPtr &msg)
{
    if(m_map_received)
    {
        pose.transform.translation.x = msg->pose.pose.position.x;
        pose.transform.translation.y = msg->pose.pose.position.y;
        pose.transform.translation.z = msg->pose.pose.position.z;
        Eigen::Vector3d vel;
        vel << msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z;
        Eigen::Quaternion<double> rot;
        Eigen::Vector3d xa; xa << 1,0,0;
        rot.setFromTwoVectors(xa, vel);
        pose.transform.rotation.w = rot.w();
        pose.transform.rotation.x = rot.x();
        pose.transform.rotation.y = rot.y();
        pose.transform.rotation.z = rot.z();
        pose.header.stamp = ros::Time::now();
        tfb.sendTransform(pose);

        int x = msg->pose.pose.position.x / m_mapInfo.resolution + m_mapTransform.getOrigin().x();
        int y = msg->pose.pose.position.y / m_mapInfo.resolution + m_mapTransform.getOrigin().y();

        int idx = 0;
        int minr = m_scan.range_min/m_mapInfo.resolution;
        int maxr = m_scan.range_max/m_mapInfo.resolution;
        for(float i=m_scan.angle_min; i<=m_scan.angle_max; i+=m_scan.angle_increment)
        {
            int x_ = x, y_ = y;
            tf2::Quaternion q;
            q.setZ(pose.transform.rotation.z);
            q.setW(pose.transform.rotation.w);
            double heading = q.getAngle() * (pose.transform.rotation.z<0 ? -1 : 1);
            float bearing = heading + i;
            m_scan.ranges[idx] = maxr;
            for(int j=minr; j<=maxr; j++)
            {
                x_ = x + j*cos(bearing);
                y_ = y + j*sin(bearing);
                if((x_<0) || (y_<0) ||
                   (x_>=m_mapData.rows()) || (y_>=m_mapData.cols()))
                {
                    m_scan.ranges[idx] = j*m_mapInfo.resolution;
                    break;
                }
                if(int(m_mapData(x_, y_)) >= 10)
                {
                    m_scan.ranges[idx] = j*m_mapInfo.resolution;
                    break;
                }
            }
            idx++;
        }

        m_scan.header.stamp = ros::Time::now();
        m_pub_scan.publish(m_scan);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pose_2_ray");
    pose_2_ray ray_casting;
    ros::spin();

    return 0;
}


