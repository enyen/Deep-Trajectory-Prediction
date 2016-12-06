#include "human_motions/prediction_merge.h"

PredictionMerge::PredictionMerge()
    : m_nh("~")
    , m_new_ls(false)
    , m_new_nn(false)
    , PREDICT_ITERATION_TIME(0.1)
{
    m_predict_step = m_nh.param("predict_time", 2.0) / PREDICT_ITERATION_TIME;
    m_boundary = m_nh.param("decision_boundary", 0.8);
    m_sub_param_nn = m_nh.subscribe("/human_traj/nn_param", 1, &PredictionMerge::handle_param_nn, this);
    m_sub_param_ls = m_nh.subscribe("/human_traj/leastsquare_param", 1, &PredictionMerge::handle_param_ls, this);
    m_pub_path_merged = m_nh.advertise<nav_msgs::Path>("/human_traj/path_merge", 1);

    std::string grid_topic = m_nh.param("occupancy_grid", std::string(""));
    if(grid_topic.empty())
    {
        ROS_ERROR_STREAM("no occupancy_grid provided. Check launch file.");
        std::abort();
    }
    m_sub_map = m_nh.subscribe(grid_topic, 1, &PredictionMerge::handle_map, this);
    m_pub_map_merged = m_nh.advertise<nav_msgs::OccupancyGrid>("human_traj/map_merged", 1);
    m_pub_map_ls = m_nh.advertise<nav_msgs::OccupancyGrid>("human_traj/map_ls", 1);
    m_pub_map_nn = m_nh.advertise<nav_msgs::OccupancyGrid>("human_traj/map_nn", 1);

    std::string path_topic = m_nh.param("path_topic", std::string("/human_traj/path_pass"));
    m_sub_path = m_nh.subscribe(path_topic, 1, &PredictionMerge::handle_path_passed, this);

    m_srv_cli_param_ls = m_nh.serviceClient<path2param>("/leastsquare/pose2params_ls");
    m_srv_cli_param_nn = m_nh.serviceClient<path2param>("/nn/pose2params_nn");
    if(!m_srv_cli_param_ls.waitForExistence(ros::Duration(10)))
        ROS_ERROR_STREAM("No ls service server");
    if(!m_srv_cli_param_nn.waitForExistence(ros::Duration(10)))
        ROS_ERROR_STREAM("No nn service server");

    ROS_INFO_STREAM("Trajectory Predictions Merging started.");
}

PredictionMerge::~PredictionMerge()
{

}

void PredictionMerge::handle_map(const nav_msgs::OccupancyGridPtr& msg)
{
    m_mapInfo = msg->info;
    m_mapData_merged = MatrixXc::Zero(m_mapInfo.width, m_mapInfo.height);
    m_mapData_ls = MatrixXc::Zero(m_mapInfo.width, m_mapInfo.height);
    m_mapData_nn = MatrixXc::Zero(m_mapInfo.width, m_mapInfo.height);

    tf::TransformListener listener;
    listener.waitForTransform("/map", "/mocap", msg->header.stamp, ros::Duration(5));
    listener.lookupTransform("/map", "/mocap", ros::Time(0), m_mapTransform);
    m_mapTransform.setOrigin(tf::Vector3(m_mapTransform.getOrigin().x()/m_mapInfo.resolution,
                                         m_mapTransform.getOrigin().y()/m_mapInfo.resolution,
                                         0));
    ROS_INFO_STREAM("Map received.");
}

void PredictionMerge::handle_path_passed(const nav_msgs::PathConstPtr &msg)
{
    m_path_pass = *msg;

    double ls_score=0, nn_score=0;
    for(int i=0; i<msg->poses.size(); i++)
    {
        int x = msg->poses[i].pose.position.x / m_mapInfo.resolution + m_mapTransform.getOrigin().x();
        int y = msg->poses[i].pose.position.y / m_mapInfo.resolution + m_mapTransform.getOrigin().y();

        ls_score += m_mapData_ls(x,y);
        nn_score += m_mapData_nn(x,y);
    }

    bool front = (m_boundary < 0.5);
    m_boundary = m_boundary*0.99 + ls_score/(ls_score+nn_score+1e-3)*0.01;
    if(!(front^(m_boundary>=0.5)))
        ROS_INFO_STREAM(((m_boundary<0.5) ? "NeuralNet" : "LeastSquare") << " dominating.");
}

void PredictionMerge::handle_param_ls(const std_msgs::Float64MultiArrayConstPtr& msg)
{
    m_new_ls = true;
    m_param_ls = msg->data;
    if(m_new_nn)
        merge();
}

void PredictionMerge::handle_param_nn(const std_msgs::Float64MultiArrayConstPtr& msg)
{
    m_new_nn = true;
    m_param_nn = msg->data;
    if(m_new_ls)
        merge();
}

void PredictionMerge::merge()
{
    m_new_ls = false;
    m_new_nn = false;

    std::vector<geometry_msgs::PoseStamped> path_predict;
//    predict_once(path_predict);
    predict_recursive(path_predict);

    m_mapData_merged = (m_mapData_merged.cast<double>() / 1.1).cast<signed char>();
    m_mapData_ls = (m_mapData_ls.cast<double>() / 1.1).cast<signed char>();
    m_mapData_nn = (m_mapData_nn.cast<double>() / 1.1).cast<signed char>();

    nav_msgs::OccupancyGrid navMap;
    navMap.info = m_mapInfo;
    navMap.header.stamp = ros::Time::now();
    navMap.header.frame_id = "map";
    navMap.data = std::vector<signed char>(m_mapData_merged.data(), m_mapData_merged.data()+m_mapData_merged.size());
    m_pub_map_merged.publish(navMap);
    navMap.data = std::vector<signed char>(m_mapData_ls.data(), m_mapData_ls.data()+m_mapData_ls.size());
    m_pub_map_ls.publish(navMap);
    navMap.data = std::vector<signed char>(m_mapData_nn.data(), m_mapData_nn.data()+m_mapData_nn.size());
    m_pub_map_nn.publish(navMap);

    nav_msgs::Path nav_path;
    nav_path.header.frame_id = "mocap";
    nav_path.poses = path_predict;
    m_pub_path_merged.publish(nav_path);
}

void PredictionMerge::predict_once(std::vector<geometry_msgs::PoseStamped>& path_predict)
{
    float span = m_param_nn[7] - m_param_nn[6];
    float span2 = 0;
    int confidence = 100;
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
        Eigen::Vector3d w_x_ls; w_x_ls << m_param_ls[0], m_param_ls[1], m_param_ls[2];
        Eigen::Vector3d w_y_ls; w_y_ls << m_param_ls[3], m_param_ls[4], m_param_ls[5];
        Eigen::Vector3d w_x_nn; w_x_nn << m_param_nn[0], m_param_nn[1], m_param_nn[2];
        Eigen::Vector3d w_y_nn; w_y_nn << m_param_nn[3], m_param_nn[4], m_param_nn[5];

        float x_ls = X_*w_x_ls, y_ls = X_*w_y_ls;
        float x_nn = X_2*w_x_nn, y_nn = X_2*w_y_nn;

        new_pose.pose.position.x = weighting*x_ls + (1.0-weighting)*x_nn;
        new_pose.pose.position.y = weighting*y_ls + (1.0-weighting)*y_nn;
        new_pose.pose.position.z = 0;
        new_pose.pose.orientation.w = 1;

        path_predict.push_back(new_pose);
        updateMap(new_pose.pose.position.x, new_pose.pose.position.y, confidence, m_mapData_merged);
        updateMap(x_ls, y_ls, confidence, m_mapData_ls);
        updateMap(x_nn, y_nn, confidence, m_mapData_nn);

        confidence = std::max(confidence-5, 0);
    }
}

void PredictionMerge::predict_recursive(std::vector<geometry_msgs::PoseStamped> &path_predict)
{
    nav_msgs::Path mypath = m_path_pass;
    path2param srv;

    float span = m_param_ls[7] - m_param_ls[6];
    float span2 = 0;
    int confidence = 100;
    for(int i=0; i<m_predict_step; i++)
    {
        double weighting = std::min(std::max(m_boundary-(i*1.f/m_predict_step) + 0.5, 0.0), 1.0);

        span += PREDICT_ITERATION_TIME;
        span2 += PREDICT_ITERATION_TIME;
        geometry_msgs::PoseStamped new_pose;

        new_pose.header.stamp = m_path_pass.poses.back().header.stamp + ros::Duration((i+1)*PREDICT_ITERATION_TIME);
        new_pose.header.frame_id = "mocap";

        Eigen::RowVector3d X_; X_ << 1.0, span, pow(span,2);
        Eigen::RowVector3d X_2; X_2 << 1.0, span2, pow(span2,2);
        Eigen::Vector3d w_x_ls; w_x_ls << m_param_ls[0], m_param_ls[1], m_param_ls[2];
        Eigen::Vector3d w_y_ls; w_y_ls << m_param_ls[3], m_param_ls[4], m_param_ls[5];
        Eigen::Vector3d w_x_nn; w_x_nn << m_param_nn[0], m_param_nn[1], m_param_nn[2];
        Eigen::Vector3d w_y_nn; w_y_nn << m_param_nn[3], m_param_nn[4], m_param_nn[5];

        float x_ls = X_*w_x_ls, y_ls = X_*w_y_ls;
        float x_nn = X_2*w_x_nn, y_nn = X_2*w_y_nn;

        new_pose.pose.position.x = weighting*x_ls + (1.0-weighting)*x_nn;
        new_pose.pose.position.y = weighting*y_ls + (1.0-weighting)*y_nn;
        new_pose.pose.position.z = 0;
        new_pose.pose.orientation.w = 1;

        path_predict.push_back(new_pose);
        updateMap(new_pose.pose.position.x, new_pose.pose.position.y, confidence, m_mapData_merged);
        updateMap(x_ls, y_ls, confidence, m_mapData_ls);
        updateMap(x_nn, y_nn, confidence, m_mapData_nn);

        confidence = std::max(confidence-5, 0);

        mypath.poses.push_back(new_pose);
        mypath.poses.erase(mypath.poses.begin());
        if((i%2==0) && (i!=0))
        {
            srv.request.path = mypath;
            if(m_srv_cli_param_ls.call(srv))
            {
                m_param_ls = srv.response.params.data;
                span = m_param_ls[7] - m_param_ls[6];
            }
            else
                ROS_ERROR_STREAM("bad ls service call");
            if(m_srv_cli_param_nn.call(srv))
            {
                m_param_nn = srv.response.params.data;
                span2 = 0;
            }
            else
                ROS_ERROR_STREAM("bad nn service call");
        }
    }
}

void PredictionMerge::updateMap(float new_pose_x, float new_pose_y, int confidence, MatrixXc& map)
{
    int x = new_pose_x / m_mapInfo.resolution + m_mapTransform.getOrigin().x();
    int y = new_pose_y / m_mapInfo.resolution + m_mapTransform.getOrigin().y();
    if((x+1<m_mapInfo.width) && (x-1>=0) && (y+1<m_mapInfo.height) && (y-1>=0))
    {
        map(x, y) = std::min(map(x, y)+confidence, 100);
        map(x+1, y) = std::min(map(x, y)+confidence/3, 100);
        map(x-1, y) = std::min(map(x, y)+confidence/3, 100);
        map(x, y+1) = std::min(map(x, y)+confidence/3, 100);
        map(x, y-1) = std::min(map(x, y)+confidence/3, 100);
        map(x+1, y+1) = std::min(map(x, y)+confidence/4, 100);
        map(x-1, y-1) = std::min(map(x, y)+confidence/4, 100);
        map(x-1, y+1) = std::min(map(x, y)+confidence/4, 100);
        map(x+1, y-1) = std::min(map(x, y)+confidence/4, 100);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "path_merge");
    PredictionMerge predictMerge;
    ros::spin();

    return 0;
}
