#include "human_motions/prediction_merge.h"

PredictionMerge::PredictionMerge()
    : m_nh("~")
    , m_new_ls(false)
    , m_new_nn(false)
    , PREDICT_ITERATION_TIME(0.1)
{
    m_predict_step = m_nh.param("predict_time", 2.0) / PREDICT_ITERATION_TIME;
    m_boundary = m_nh.param("decision_boundary", 0.5);
    m_sub_nn = m_nh.subscribe("/human_traj/nn_param", 1, &PredictionMerge::handle_path_nn, this);
    m_sub_ls = m_nh.subscribe("/human_traj/leastsquare_param", 1, &PredictionMerge::handle_path_ls, this);
    m_pub_path = m_nh.advertise<nav_msgs::Path>("/human_traj/path_merge", 1);
}

PredictionMerge::~PredictionMerge()
{

}

void PredictionMerge::handle_path_ls(const std_msgs::Float64MultiArrayConstPtr& msg)
{
    m_new_ls = true;
    m_path_ls = msg->data;
    if(m_new_nn)
        merge();
}

void PredictionMerge::handle_path_nn(const std_msgs::Float64MultiArrayConstPtr& msg)
{
    m_new_nn = true;
    m_path_nn = msg->data;
    if(m_new_ls)
        merge();
}

void PredictionMerge::merge()
{
    m_new_ls = false;
    m_new_nn = false;

    std::vector<geometry_msgs::PoseStamped> path_predict;
    float span = m_path_nn[7] - m_path_nn[6];
    float span2 = 0;
    for(int i=0; i<m_predict_step; i++)
    {
        double weighting = std::min(std::max(m_boundary-(i*1.f/m_predict_step) + 0.5, 0.0), 1.0);

        span += PREDICT_ITERATION_TIME;
        span2 += PREDICT_ITERATION_TIME;
        geometry_msgs::PoseStamped new_pose;
        new_pose.header.stamp += ros::Duration(span);
        new_pose.header.frame_id = "mocap";

        Eigen::RowVector3d X_; X_ << 1.0, span, pow(span,2);
        Eigen::RowVector3d X_2; X_2 << 1.0, span2, pow(span2,2);
        Eigen::Vector3d w_x_ls; w_x_ls << m_path_ls[0], m_path_ls[1], m_path_ls[2];
        Eigen::Vector3d w_y_ls; w_y_ls << m_path_ls[3], m_path_ls[4], m_path_ls[5];
        Eigen::Vector3d w_x_nn; w_x_nn << m_path_nn[0], m_path_nn[1], m_path_nn[2];
        Eigen::Vector3d w_y_nn; w_y_nn << m_path_nn[3], m_path_nn[4], m_path_nn[5];

        new_pose.pose.position.x = weighting*X_*w_x_ls + (1.0-weighting)*X_2.dot(w_x_nn);
        new_pose.pose.position.y = weighting*X_*w_y_ls + (1.0-weighting)*X_2.dot(w_y_nn);
        new_pose.pose.position.z = 0;
        new_pose.pose.orientation.w = 1;

        path_predict.push_back(new_pose);
    }

    nav_msgs::Path nav_path;
    nav_path.header.frame_id = "mocap";
    nav_path.poses = path_predict;
    m_pub_path.publish(nav_path);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "path_merge");
    PredictionMerge predictMerge;
    ros::spin();

    return 0;
}
