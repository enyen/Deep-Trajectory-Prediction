#ifndef PREDICTIONMERGE_H
#define PREDICTIONMERGE_H

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/MapMetaData.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf/transform_listener.h>
#include <human_motions/path2params.h>

#include <eigen3/Eigen/Dense>
#include <vector>

class PredictionMerge
{
    typedef Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic> MatrixXc;

public:
    PredictionMerge();
    ~PredictionMerge();

    inline bool new_ls(){
        return m_new_ls;
    }
    inline bool new_nn(){
        return m_new_nn;
    }

private:
    ros::NodeHandle m_nh;
    ros::Subscriber m_sub_param_nn;
    ros::Subscriber m_sub_param_ls;
    ros::Publisher m_pub_path_merged;
    ros::Subscriber m_sub_map;
    ros::Publisher m_pub_map_merged;
    ros::Publisher m_pub_map_ls;
    ros::Publisher m_pub_map_nn;
    ros::Subscriber m_sub_path;
    using path2param = human_motions::path2params;
    ros::ServiceClient m_srv_cli_param_ls;
    ros::ServiceClient m_srv_cli_param_nn;

    void handle_param_nn(const std_msgs::Float64MultiArrayConstPtr& msg);
    void handle_param_ls(const std_msgs::Float64MultiArrayConstPtr& msg);
    void handle_map(const nav_msgs::OccupancyGridPtr& msg);
    void handle_path_passed(const nav_msgs::PathConstPtr& msg);
    void merge();
    void updateMap(float new_pose_x, float new_pose_y, int confidence, MatrixXc& map);
    void predict_once(std::vector<geometry_msgs::PoseStamped>& path_predict);
    void predict_recursive(std::vector<geometry_msgs::PoseStamped>& path_predict);

    std::vector<double> m_param_nn;
    std::vector<double> m_param_ls;
    nav_msgs::MapMetaData m_mapInfo;
    tf::StampedTransform m_mapTransform;
    MatrixXc m_mapData_merged;
    MatrixXc m_mapData_ls;
    MatrixXc m_mapData_nn;
    nav_msgs::Path m_path_pass;

    bool m_new_nn;
    bool m_new_ls;
    int m_predict_step;
    float m_boundary;
    const float PREDICT_ITERATION_TIME;
};

#endif // PREDICTIONMERGE_H
