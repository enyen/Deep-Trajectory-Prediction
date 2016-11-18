#ifndef PREDICTIONMERGE_H
#define PREDICTIONMERGE_H

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64MultiArray.h>
#include <eigen3/Eigen/Dense>
#include <vector>

class PredictionMerge
{
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
    ros::Subscriber m_sub_nn;
    ros::Subscriber m_sub_ls;
    ros::Publisher m_pub_path;

    void handle_path_nn(const std_msgs::Float64MultiArrayConstPtr& msg);
    void handle_path_ls(const std_msgs::Float64MultiArrayConstPtr& msg);
    void merge();

    std::vector<double> m_path_nn;
    std::vector<double> m_path_ls;
    bool m_new_nn;
    bool m_new_ls;
    int m_predict_step;
    float m_boundary;
    const float PREDICT_ITERATION_TIME;
};

#endif // PREDICTIONMERGE_H
