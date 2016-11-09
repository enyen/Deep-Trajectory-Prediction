#include "human_motions/human_model.h"
#include <geometry_msgs/Twist.h>

HumanModel::HumanModel()
    : m_nh("~")
{
    m_path_size = m_nh.param("path_size", 20);
    m_predict_step = m_nh.param("predict_time", 2.0) / (0.1);

    std::string odom_topic = m_nh.param("odom_topic", std::string(""));
    if(odom_topic.empty())
    {
        ROS_INFO_STREAM("no odom_topic provided. Check launch file.");
        std::abort();
    }
    m_sub_odom= m_nh.subscribe(odom_topic, 1, &HumanModel::handleOdom, this);
//    m_sub_pose= m_nh.subscribe("/mocap/obs_pose", 2000, &HumanModel::handlePose, this);

//    m_pub_odom_vel = m_nh.advertise<geometry_msgs::PoseStamped>("/human_pose/odom_vel",1);
    m_pub_path_pass = m_nh.advertise<nav_msgs::Path>("/human_pose/path_pass",1);
    m_pub_path_predict = m_nh.advertise<nav_msgs::Path>("/human_pose/path_predicted",1);

//    m_myfile.open("/home/enyen/Documents/tranjactory4.txt");
//    m_time_start = -1;
//    count = 0;
}

HumanModel::~HumanModel()
{
//    m_myfile.close();
}

void HumanModel::handleOdom(const nav_msgs::OdometryConstPtr &msg)
{
    // path current velocity
//    geometry_msgs::PoseStamped vel_pose;
//    vel_pose.header = msg->header;

//    vel_pose.pose.position = msg->pose.pose.position;
//    Eigen::Vector3d vel;
//    vel << msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z;
//    Eigen::Quaternion<double> rot;
//    Eigen::Vector3d xa; xa << 1,0,0;
//    rot.setFromTwoVectors(xa, vel);
//    vel_pose.pose.orientation.w = rot.w();
//    vel_pose.pose.orientation.x = rot.x();
//    vel_pose.pose.orientation.y = rot.y();
//    vel_pose.pose.orientation.z = rot.z();

//    m_pub_odom_vel.publish(vel_pose);

    // path prediction
    if(m_path_Y.size() >= m_path_size)
    {
        m_path_Y.erase(m_path_Y.begin());
        m_path_pose.erase(m_path_pose.begin());
    }
    geometry_msgs::PoseStamped new_pose;
    new_pose.header = msg->header;
    new_pose.pose = msg->pose.pose;
    m_path_pose.push_back(new_pose);

    m_path_X.clear();
    for(int i=0; i<m_path_pose.size(); i++)
    {
        ros::Duration span = m_path_pose[i].header.stamp - m_path_pose[0].header.stamp;
        double x_span = span.toSec();
        m_path_X.push_back((Eigen::Vector3f() <<
                            1.0,
                            x_span,
                            pow(x_span,2)
                           ).finished());
    }
    m_path_Y.push_back((Eigen::Vector2f()<<
                        msg->pose.pose.position.x,
                        msg->pose.pose.position.y
                       ).finished());

    Eigen::Map<MatrixXf_rowMajor> X(m_path_X[0].data(), (int)m_path_X.size(), 3);
    Eigen::Map<MatrixXf_rowMajor> Y(m_path_Y[0].data(), (int)m_path_X.size(), 2);
    Eigen::MatrixXf tempM = (X.transpose()*X).ldlt().solve(X.transpose());
    Eigen::VectorXf weights_x = tempM * Y.col(0);
    Eigen::VectorXf weights_y = tempM * Y.col(1);

    std::vector<geometry_msgs::PoseStamped> path_predict;
    ros::Duration span_predict = msg->header.stamp - m_path_pose[0].header.stamp;
    for(int i=0; i<m_predict_step; i++)
    {
        span_predict += ros::Duration(0.1);
        geometry_msgs::PoseStamped new_pose;
        new_pose.header = msg->header;

        double x_span = span_predict.toSec();
        Eigen::RowVector3f X_; X_ << 1.0, x_span, pow(x_span,2);
        new_pose.pose.position.x = X_ * weights_x;
        new_pose.pose.position.y = X_ * weights_y;
        new_pose.pose.position.z = msg->pose.pose.position.z;
        new_pose.pose.orientation = msg->pose.pose.orientation;

        path_predict.push_back(new_pose);
    }

    nav_msgs::Path nav_path;
    nav_path.header = msg->header;
    nav_path.poses = m_path_pose;
    m_pub_path_pass.publish(nav_path);
    nav_path.poses = path_predict;
    m_pub_path_predict.publish(nav_path);
}

//void HumanModel::handlePose(const geometry_msgs::PoseStampedConstPtr &msg)
//{
//    if(count%100)ROS_INFO_STREAM("count "<<count);
//    count++;

//    if(m_time_start == -1) m_time_start = msg->header.stamp.toSec();

//    m_myfile << patch::to_string(msg->header.stamp.toSec() - m_time_start);
//    m_myfile << " ";
//    m_myfile << patch::to_string(msg->pose.position.x);
//    m_myfile << " ";
//    m_myfile << patch::to_string(msg->pose.position.y);
//    m_myfile << "\n";

////    m_time_start = msg->header.stamp.toSec();
//}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "human_pose");
    HumanModel model;
    ros::spin();

    return 0;
}
