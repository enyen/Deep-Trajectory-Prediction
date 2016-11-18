#ifndef LEASTSQUARE_PREDICT_H
#define LEASTSQUARE_PREDICT_H

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64MultiArray.h>
#include <eigen3/Eigen/Dense>
#include <vector>

class LeastSquare_Predict
{
public:
    LeastSquare_Predict();
    ~LeastSquare_Predict();

private:
    ros::NodeHandle m_nh;
    ros::Subscriber m_sub_path;
    ros::Publisher m_pub_path;
    ros::Publisher m_pub_param;

    void handle_path(const nav_msgs::PathConstPtr& msg);

    int m_predict_step;
    const float PREDICT_ITERATION_TIME;
};

#endif // LEASTSQUARE_PREDICT_H
