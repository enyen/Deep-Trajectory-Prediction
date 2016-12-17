#include "human_motions/leastsquare_predict.h"

LeastSquare_Predict::LeastSquare_Predict()
    : m_nh("~")
    , PREDICT_ITERATION_TIME(0.1)
{
    std::string path_topic = m_nh.param("path_topic", std::string("/human_traj/path_pass"));
    m_predict_step = ceil(m_nh.param("predict_time", 2.0) / PREDICT_ITERATION_TIME);
    m_sub_path = m_nh.subscribe(path_topic, 1, &LeastSquare_Predict::handle_path, this);
    m_pub_path = m_nh.advertise<nav_msgs::Path>("/human_traj/path_leastsquare", 1);
    m_pub_param = m_nh.advertise<std_msgs::Float64MultiArray>("/human_traj/leastsquare_param", 1);
    m_srv_param = m_nh.advertiseService("pose2params_ls", &LeastSquare_Predict::handle_path2params, this);

    std::string grid_topic = m_nh.param("occupancy_grid", std::string("/map"));
    m_sub_map = m_nh.subscribe(grid_topic, 1, &LeastSquare_Predict::handle_map, this);
    m_pub_map = m_nh.advertise<nav_msgs::OccupancyGrid>("human_traj/score_map_ls", 1);
    m_pub_performance = m_nh.advertise<std_msgs::Float64MultiArray>("human_traj/score_ls", 1);
    m_score = 0;
    m_map_received = false;

    ROS_INFO_STREAM("Least Square Trajectory Prediction started.");
}

LeastSquare_Predict::~LeastSquare_Predict()
{

}

void LeastSquare_Predict::handle_map(const nav_msgs::OccupancyGridConstPtr &msg)
{
    if(m_map_received)
        return;

    m_mapInfo = msg->info;
    m_mapData = MatrixXc::Zero(m_mapInfo.width, m_mapInfo.height);

    tf::TransformListener listener;
    listener.waitForTransform("/map", "/mocap", msg->header.stamp, ros::Duration(5));
    listener.lookupTransform("/map", "/mocap", ros::Time(0), m_mapTransform);
    m_mapTransform.setOrigin(tf::Vector3(m_mapTransform.getOrigin().x()/m_mapInfo.resolution,
                                         m_mapTransform.getOrigin().y()/m_mapInfo.resolution,
                                         0));
    m_map_received = true;
    ROS_INFO_STREAM("Map received.");
}

void LeastSquare_Predict::handle_path(const nav_msgs::PathConstPtr &msg)
{
    if(!m_map_received)
        return;

    double predict_score = 0;
    for(int i=0; i<msg->poses.size(); i++)
    {
        int x = msg->poses[i].pose.position.x / m_mapInfo.resolution + m_mapTransform.getOrigin().x();
        int y = msg->poses[i].pose.position.y / m_mapInfo.resolution + m_mapTransform.getOrigin().y();
        predict_score += m_mapData(x,y);
    }
    float smoother = 0.99;
    m_score = smoother*m_score + (1-smoother)*predict_score;
    std_msgs::Float64MultiArray performances;
    performances.data = {m_score};
    m_pub_performance.publish(performances);

    std_msgs::Float64MultiArray param = predict_once(*msg);
    m_pub_param.publish(param);
//    predict_recursive(msg);
}

std_msgs::Float64MultiArray LeastSquare_Predict::predict_once(const nav_msgs::Path &msg, bool pub)
{
    Eigen::MatrixXf X = Eigen::MatrixXf::Constant(msg.poses.size(), 3, 0);
    Eigen::MatrixXf Y = Eigen::MatrixXf::Constant(msg.poses.size(), 2, 0);
    for(int i=0; i<msg.poses.size(); i++)
    {
        double x_span = (msg.poses[i].header.stamp - msg.poses[0].header.stamp).toSec();
        X.row(i) = (Eigen::RowVector3f() << 1.0, x_span, pow(x_span,2)).finished();
        Y.row(i) = (Eigen::RowVector2f() << msg.poses[i].pose.position.x, msg.poses[i].pose.position.y).finished();
    }

    Eigen::MatrixXf tempM = (X.transpose()*X).ldlt().solve(X.transpose());
    Eigen::VectorXf weights_x = tempM * Y.col(0);
    Eigen::VectorXf weights_y = tempM * Y.col(1);

    std::vector<geometry_msgs::PoseStamped> path_predict;
    ros::Duration span_predict = msg.poses.back().header.stamp - msg.poses[0].header.stamp;
    for(int i=0; i<m_predict_step; i++)
    {
        span_predict += ros::Duration(PREDICT_ITERATION_TIME);
        geometry_msgs::PoseStamped new_pose;
        new_pose.header = msg.header;
        new_pose.header.stamp = msg.poses[0].header.stamp + span_predict;

        double x_span = span_predict.toSec();
        Eigen::RowVector3f X_; X_ << 1.0, x_span, pow(x_span,2);
        new_pose.pose.position.x = X_ * weights_x;
        new_pose.pose.position.y = X_ * weights_y;
        new_pose.pose.position.z = 0;
        new_pose.pose.orientation = msg.poses[i].pose.orientation;

        path_predict.push_back(new_pose);
    }

    std_msgs::Float64MultiArray param;
    param.data = {weights_x(0), weights_x(1), weights_x(2),
                  weights_y(0), weights_y(1), weights_y(2),
                  msg.poses[0].header.stamp.toSec(),
                  msg.poses.back().header.stamp.toSec()};

    if(pub)
    {
        nav_msgs::Path nav_path;
        nav_path.header = msg.header;
        nav_path.poses = path_predict;
        m_pub_path.publish(nav_path);
        update_map(path_predict);
    }

    return param;
}

void LeastSquare_Predict::predict_recursive(const nav_msgs::PathConstPtr &msg)
{
    Eigen::MatrixXf X = Eigen::MatrixXf::Constant(msg->poses.size()+m_predict_step, 3, 0);
    Eigen::MatrixXf Y = Eigen::MatrixXf::Constant(msg->poses.size()+m_predict_step, 2, 0);
    for(int i=0; i<msg->poses.size(); i++)
    {
        double x_span = (msg->poses[i].header.stamp - msg->poses[0].header.stamp).toSec();
        X.row(i) = (Eigen::RowVector3f() << 1.0, x_span, pow(x_span,2)).finished();
        Y.row(i) = (Eigen::RowVector2f() << msg->poses[i].pose.position.x, msg->poses[i].pose.position.y).finished();
    }

    std::vector<geometry_msgs::PoseStamped> path_predict;
    ros::Duration span_predict = msg->poses.back().header.stamp - msg->poses[0].header.stamp;
    int smallstep = 2;
    for(int k=0; k<m_predict_step; k+=smallstep)
    {
        Eigen::MatrixXf tempM = (X.topRows(msg->poses.size()+k).transpose()*X.topRows(msg->poses.size()+k)).ldlt().solve(X.topRows(msg->poses.size()+k).transpose());
        Eigen::VectorXf weights_x = tempM * Y.topRows(msg->poses.size()+k).col(0);
        Eigen::VectorXf weights_y = tempM * Y.topRows(msg->poses.size()+k).col(1);
        for(int i=0; i<smallstep; i++)
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
            new_pose.pose.orientation = msg->poses.back().pose.orientation;

            path_predict.push_back(new_pose);
            X.row(msg->poses.size()+k+i) = (Eigen::RowVector3f() << 1.0, x_span, pow(x_span,2)).finished();
            Y.row(msg->poses.size()+k+i) = (Eigen::RowVector2f() << new_pose.pose.position.x, new_pose.pose.position.y).finished();
        }
    }

    Eigen::MatrixXf tempM = (X.transpose()*X).ldlt().solve(X.transpose());
    Eigen::VectorXf weights_x = tempM * Y.col(0);
    Eigen::VectorXf weights_y = tempM * Y.col(1);

//    std_msgs::Float64MultiArray param;
//    param.data = {weights_x(0), weights_x(1), weights_x(2),
//                  weights_y(0), weights_y(1), weights_y(2),
//                  msg->poses[0].header.stamp.toSec(),
//                  msg->poses.back().header.stamp.toSec()};
//    m_pub_param.publish(param);

    nav_msgs::Path nav_path;
    nav_path.header = msg->header;
    nav_path.poses = path_predict;
    m_pub_path.publish(nav_path);
}

void LeastSquare_Predict::update_map(std::vector<geometry_msgs::PoseStamped> &path_predict)
{
    int confidence = 0;
    for(int i=0; i< path_predict.size(); i++)
    {
        int x = path_predict[i].pose.position.x / m_mapInfo.resolution + m_mapTransform.getOrigin().x();
        int y = path_predict[i].pose.position.y / m_mapInfo.resolution + m_mapTransform.getOrigin().y();
        if((x+1<m_mapInfo.width) && (x-1>=0) && (y+1<m_mapInfo.height) && (y-1>=0))
        {
            confidence = std::min(confidence+3, 100);
            m_mapData(x, y) = std::min(m_mapData(x, y)+confidence, 100);
            m_mapData(x+1, y) = std::min(m_mapData(x+1, y)+confidence/3, 100);
            m_mapData(x-1, y) = std::min(m_mapData(x-1, y)+confidence/3, 100);
            m_mapData(x, y+1) = std::min(m_mapData(x, y+1)+confidence/3, 100);
            m_mapData(x, y-1) = std::min(m_mapData(x, y-1)+confidence/3, 100);
            m_mapData(x+1, y+1) = std::min(m_mapData(x+1, y+1)+confidence/4, 100);
            m_mapData(x-1, y-1) = std::min(m_mapData(x-1, y-1)+confidence/4, 100);
            m_mapData(x-1, y+1) = std::min(m_mapData(x+1, y-1)+confidence/4, 100);
            m_mapData(x+1, y-1) = std::min(m_mapData(x-1, y+1)+confidence/4, 100);
        }
    }
    m_mapData = (m_mapData.cast<double>() / 1.1).cast<signed char>();
    nav_msgs::OccupancyGrid navMap;
    navMap.info = m_mapInfo;
    navMap.header.stamp = ros::Time::now();
    navMap.header.frame_id = "map";
    navMap.data = std::vector<signed char>(m_mapData.data(), m_mapData.data()+m_mapData.size());
    m_pub_map.publish(navMap);
}

bool LeastSquare_Predict::handle_path2params(path2params::Request &req, path2params::Response &res)
{
    res.params = predict_once(req.path, 0);
    return true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "leastsquare_predict");
    LeastSquare_Predict lq_predict;
    ros::spin();

    return 0;
}
