#ifndef HUMAN_MODEL_H
#define HUMAN_MODEL_H

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <eigen3/Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <sstream>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

class HumanModel
{
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rowMajor;

public:
    HumanModel();
    ~HumanModel();

private:
    void handleOdom(const nav_msgs::OdometryConstPtr& msg);
    void handlePose(const geometry_msgs::PoseStampedConstPtr& msg);

    ros::NodeHandle m_nh;
    ros::Subscriber m_sub_odom;
//    ros::Subscriber m_sub_pose;
//    ros::Publisher m_pub_odom_vel;
    ros::Publisher m_pub_path_pass;
    ros::Publisher m_pub_path_predict;

    unsigned int m_path_size;
    double m_predict_step;
    std::vector<Eigen::Vector3f> m_path_X;
    std::vector<Eigen::Vector2f> m_path_Y;
    std::vector<geometry_msgs::PoseStamped> m_path_pose;

//    std::ofstream m_myfile;
//    double m_time_start;
//    unsigned int count;
};

#endif // HUMAN_MODEL_H
