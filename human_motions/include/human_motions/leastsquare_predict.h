#ifndef LEASTSQUARE_PREDICT_H
#define LEASTSQUARE_PREDICT_H

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/Float64MultiArray.h>
#include <human_motions/path2params.h>
#include <tf/transform_listener.h>
#include <eigen3/Eigen/Dense>
#include <vector>

class LeastSquare_Predict
{
    typedef Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic> MatrixXc;

public:
    LeastSquare_Predict();
    ~LeastSquare_Predict();

private:
    ros::NodeHandle m_nh;
    ros::Subscriber m_sub_path;
    ros::Publisher m_pub_path;
    ros::Publisher m_pub_param;
    ros::Subscriber m_sub_map;
    ros::Publisher m_pub_map;
    ros::Publisher m_pub_performance;
    using path2params = human_motions::path2params;
    ros::ServiceServer m_srv_param;

    void handle_path(const nav_msgs::PathConstPtr& msg);
    void handle_map(const nav_msgs::OccupancyGridConstPtr& msg);
    void predict_recursive(const nav_msgs::PathConstPtr &msg);
    std_msgs::Float64MultiArray predict_once(const nav_msgs::Path &msg, bool pub=1);
    bool handle_path2params(path2params::Request &req, path2params::Response &res);
    void update_map(std::vector<geometry_msgs::PoseStamped>& path_predict);

    int m_predict_step;
    const float PREDICT_ITERATION_TIME;
    double m_score;
    bool m_map_received;
    nav_msgs::MapMetaData m_mapInfo;
    tf::StampedTransform m_mapTransform;
    MatrixXc m_mapData;
};

#endif // LEASTSQUARE_PREDICT_H
