#include "pose_2_ray.h"

#include <tf/transform_broadcaster.h>

pose_2_ray::pose_2_ray()
    : m_nh("~")
    , range(0.2)
{
    std::string map_topic = m_nh.param("occupancy_map_topic", std::string(""));
    if(map_topic.empty())
    {
        ROS_INFO_STREAM("no occupancy_map_topic provided. Check launch file.");
        std::abort();
    }
    std::string odom_topic = m_nh.param("odom_topic", std::string(""));
    if(odom_topic.empty())
    {
        ROS_INFO_STREAM("no odom_topic provided. Check launch file.");
        std::abort();
    }
    m_sub_map = m_nh.subscribe(map_topic, 1, &pose_2_ray::handleMap, this);
    m_sub_odom = m_nh.subscribe(odom_topic, 1, &pose_2_ray::handleOdom, this);
    m_sub_mocap = m_nh.subscribe("/mocap", 1, &pose_2_ray::handleMocap, this);

    m_pub_scan = m_nh.advertise<sensor_msgs::LaserScan>("/pose_2_ray/scan", 1);

    m_scan.angle_min = -M_PI;
    m_scan.angle_max = M_PI;
    m_scan.angle_increment = 2.0*M_PI / 179;
    m_scan.range_max = 30;
    m_scan.range_min = 0.01;

    asd = m_nh.advertise<nav_msgs::OccupancyGrid>("/asd", 1);
}

pose_2_ray::~pose_2_ray()
{
}

void pose_2_ray::handleMap(const nav_msgs::OccupancyGridConstPtr &msg)
{
    m_map = *msg;
//    m_map.info.resolution = 0.1;
//    asd.publish(m_map);
}

void pose_2_ray::handleOdom(const nav_msgs::OdometryConstPtr &msg)
{
    if(m_map.data.size())
    {
        nav_msgs::OccupancyGrid new_occ_map;
        new_occ_map.header = m_map.header;
        new_occ_map.info = m_map.info;
        Eigen::Map<MatrixXi_rowMajor> new_map(m_map.data.data(), m_map.info.width, m_map.info.height);
        Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> new_map_mat = new_map;

//        double x = 0, y = -1;
        double x = msg->pose.pose.position.x, y = msg->pose.pose.position.y;

        new_map_mat.conservativeResize(Eigen::NoChange, m_map.info.height+abs(2*y/m_map.info.resolution));
        new_occ_map.info.height += abs(2*y/m_map.info.resolution);
        if(y > 0)
        {
            new_map_mat.rightCols(abs(2*y/m_map.info.resolution)) =
                    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>::Constant(
                        new_map_mat.rows(), abs(2*y/m_map.info.resolution), -1);
        }
        else if(y < 0)
        {
            Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> tempcopy = new_map_mat.leftCols(m_map.info.height);
            new_map_mat.rightCols(m_map.info.height) = tempcopy;
            new_map_mat.leftCols(abs(2*y/m_map.info.resolution)) =
                    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>::Constant(
                        new_map_mat.rows(), abs(2*y/m_map.info.resolution), -1);
        }

        new_map_mat.conservativeResize(m_map.info.width+abs(2*x/m_map.info.resolution), Eigen::NoChange);
        new_occ_map.info.width += abs(2*x/m_map.info.resolution);
        if(x > 0)
        {
            new_map_mat.bottomRows(abs(2*x/m_map.info.resolution)) =
                    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>::Constant(
                        abs(2*x/m_map.info.resolution), new_map_mat.cols(), -1);
        }else if(x < 0)
        {
            Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> tempcopy = new_map_mat.topRows(m_map.info.width);
            new_map_mat.bottomRows(m_map.info.width) = tempcopy;
            new_map_mat.topRows(abs(2*x/m_map.info.resolution)) =
                    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>::Constant(
                        abs(2*x/m_map.info.resolution), new_map_mat.cols(), -1);
        }

//        new_map_mat.block<100, 50>(40,30) =
//                Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>::Constant(100, 50, 0);

        new_occ_map.data = std::vector<int8_t>(new_map_mat.data(), new_map_mat.data()+new_map_mat.size());

//        tf::TransformBroadcaster tb;
//        tf::Transform laser_frame;
//        laser_frame.setOrigin(tf::Vector3(msg->pose.pose.position.x, msg->pose.pose.position.y, 0));
//        tf::Quaternion q; q.setRPY(0, 0, 0);
//        laser_frame.setRotation(q);
//        tb.sendTransform(tf::StampedTransform(laser_frame,
//                                              msg->header.stamp+ros::Duration(0.1),
//                                              "mocap",
//                                              "laser_frame"));
        m_scan.header = msg->header;
        m_scan.header.frame_id = "/mocap";

        map_ray_caster::MapRayCaster m_ray_caster;
        asd.publish(new_occ_map);

        m_ray_caster.laserScanCast(new_occ_map, m_scan);
        m_pub_scan.publish(m_scan);
    }
}

void pose_2_ray::handleMocap(const geometry_msgs::TransformConstPtr &msg)
{
//    tf::transformMsgToTF()
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "pose_2_ray");
    pose_2_ray ray_casting;
    ros::spin();

    return 0;
}


