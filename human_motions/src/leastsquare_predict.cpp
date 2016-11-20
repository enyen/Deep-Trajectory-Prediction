#include "human_motions/leastsquare_predict.h"

LeastSquare_Predict::LeastSquare_Predict()
    : m_nh("~")
    , PREDICT_ITERATION_TIME(0.1)
{
    std::string path_topic = m_nh.param("path_topic", std::string("/human_traj/path_pass"));
    m_predict_step = m_nh.param("predict_time", 2.0) / PREDICT_ITERATION_TIME;
    m_sub_path = m_nh.subscribe(path_topic, 1, &LeastSquare_Predict::handle_path, this);
    m_pub_path = m_nh.advertise<nav_msgs::Path>("/human_traj/path_leastsquare", 1);
    m_pub_param = m_nh.advertise<std_msgs::Float64MultiArray>("/human_traj/leastsquare_param", 1);

    ROS_INFO_STREAM("Least Square Trajectory Prediction started.");
}

LeastSquare_Predict::~LeastSquare_Predict()
{

}

void LeastSquare_Predict::handle_path(const nav_msgs::PathConstPtr &msg)
{
    Eigen::MatrixXf X = Eigen::MatrixXf::Constant(msg->poses.size(), 3, 0);
    Eigen::MatrixXf Y = Eigen::MatrixXf::Constant(msg->poses.size(), 2, 0);
    for(int i=0; i<msg->poses.size(); i++)
    {
        double x_span = (msg->poses[i].header.stamp - msg->poses[0].header.stamp).toSec();
        X.row(i) = (Eigen::RowVector3f() << 1.0, x_span, pow(x_span,2)).finished();
        Y.row(i) = (Eigen::RowVector2f() << msg->poses[i].pose.position.x, msg->poses[i].pose.position.y).finished();
    }

    Eigen::MatrixXf tempM = (X.transpose()*X).ldlt().solve(X.transpose());
    Eigen::VectorXf weights_x = tempM * Y.col(0);
    Eigen::VectorXf weights_y = tempM * Y.col(1);

    std_msgs::Float64MultiArray param;
    param.data = {weights_x(0), weights_x(1), weights_x(2),
                  weights_y(0), weights_y(1), weights_y(2),
                  msg->poses[0].header.stamp.toSec(),
                  msg->poses[msg->poses.size()-1].header.stamp.toSec()};
    m_pub_param.publish(param);

    std::vector<geometry_msgs::PoseStamped> path_predict;
    ros::Duration span_predict = msg->poses[msg->poses.size()-1].header.stamp - msg->poses[0].header.stamp;
    for(int i=0; i<m_predict_step; i++)
    {
        span_predict += ros::Duration(PREDICT_ITERATION_TIME);
        geometry_msgs::PoseStamped new_pose;
        new_pose.header = msg->header;
        new_pose.header.stamp = msg->poses[0].header.stamp + span_predict;

        double x_span = span_predict.toSec();
        Eigen::RowVector3f X_; X_ << 1.0, x_span, pow(x_span,2);
        new_pose.pose.position.x = X_ * weights_x;
        new_pose.pose.position.y = X_ * weights_y;
        new_pose.pose.position.z = 0;
        new_pose.pose.orientation = msg->poses[i].pose.orientation;

        path_predict.push_back(new_pose);
    }

    nav_msgs::Path nav_path;
    nav_path.header = msg->header;
    nav_path.poses = path_predict;
    m_pub_path.publish(nav_path);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "leastsquare_predict");
    LeastSquare_Predict lq_predict;
    ros::spin();

    return 0;
}
