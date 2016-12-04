#include "local_planner.h"
#include <pluginlib/class_list_macros.h>


LocalPlanner::LocalPlanner()
    : m_nh("~")
{
    std::string obstacle_map = m_nh.param("obstacle_map", std::string("human_traj/map_merged"));
    m_sub_map_obstacle = m_nh.subscribe(obstacle_map, 1, &LocalPlanner::handle_map_obstacle, this);
    std::string global_map = m_nh.param("local_map", std::string("/move_base/global_costmap/costmap"));
    m_sub_map_global = m_nh.subscribe(global_map, 1, &LocalPlanner::handle_map_global, this);

    asd = m_nh.advertise<nav_msgs::OccupancyGrid>("asd", 1);
}

LocalPlanner::~LocalPlanner()
{

}

void LocalPlanner::handle_map_obstacle(const nav_msgs::OccupancyGridConstPtr &msg)
{
    m_map_obstacle = *msg;
    Eigen::Map<MatrixXi8> map(m_map_obstacle.data.data(), m_map_obstacle.info.width, m_map_obstacle.info.height);
    m_mat_obstacle = map;
}

void LocalPlanner::handle_map_global(const nav_msgs::OccupancyGridConstPtr &msg)
{
    m_map_global = *msg;
    Eigen::Map<MatrixXi8> map(m_map_global.data.data(), m_map_global.info.width, m_map_global.info.height);
    m_mat_global = map;
}

void LocalPlanner::initialize(std::string name,
                              tf::TransformListener *tf,
                              costmap_2d::Costmap2DROS *costmap_ros)
{
    m_tf = tf;
    m_costmap_ros = costmap_ros;
}

bool LocalPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
{
    cmd_vel.linear.x = 0.1;
    cmd_vel.angular.z = 0.2;

    Eigen::Map<MatrixXi8> map(m_map_gpath.data.data(), m_map_gpath.info.width, m_map_gpath.info.height);
    MatrixXi8 m_mat_gpath = map;
    nav_msgs::OccupancyGrid map_local;
    map_local.header.stamp = ros::Time::now();
    map_local.header.frame_id = "map";
    map_local.info.height = 60;
    map_local.info.width = 60;
    map_local.info.resolution = 0.05;
    tf::Stamped<tf::Pose> robotpose;
    m_costmap_ros->getRobotPose(robotpose);
    map_local.info.origin.position.x = robotpose.getOrigin().getX()-1.5;
    map_local.info.origin.position.y = robotpose.getOrigin().getY()-1.5;
    map_local.info.origin.orientation.w = 1;
    MatrixXi8 mat = m_mat_gpath.block<60,60>(robotpose.getOrigin().getX()*1.0/0.05-30, robotpose.getOrigin().getY()*1.0/0.05-30);
    map_local.data = std::vector<signed char>(mat.data(), mat.data()+mat.size());
    asd.publish(map_local);

    return true;
}

bool LocalPlanner::isGoalReached()
{
    return false;
}

bool LocalPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan)
{
    ROS_INFO_STREAM("Plan size: " << plan.size());
    m_map_gpath = m_map_global;
    int dist = -300;
    for(int i=plan.size()-1; i>=0; i--)
    {
        int x = plan[i].pose.position.x / m_map_gpath.info.resolution;
        int y = plan[i].pose.position.y / m_map_gpath.info.resolution;
        m_map_gpath.data[x+y*m_map_gpath.info.width] = m_map_gpath.data[x+y*m_map_gpath.info.width]<dist ? m_map_gpath.data[x+y*m_map_gpath.info.width]:dist;
        m_map_gpath.data[x+1+y*m_map_gpath.info.width] = m_map_gpath.data[x+1+y*m_map_gpath.info.width]<dist/2 ? m_map_gpath.data[x+1+y*m_map_gpath.info.width]:dist/2;
        m_map_gpath.data[x-1+y*m_map_gpath.info.width] = m_map_gpath.data[x-1+y*m_map_gpath.info.width]<dist/2 ? m_map_gpath.data[x-1+y*m_map_gpath.info.width]:dist/2;
        m_map_gpath.data[x+(y+1)*m_map_gpath.info.width] = m_map_gpath.data[x+(y+1)*m_map_gpath.info.width]<dist/2 ? m_map_gpath.data[x+(y+1)*m_map_gpath.info.width]:dist/2;
        m_map_gpath.data[x+(y-1)*m_map_gpath.info.width] = m_map_gpath.data[x+(y-1)*m_map_gpath.info.width]<dist/2 ? m_map_gpath.data[x+(y-1)*m_map_gpath.info.width]:dist/2;
        m_map_gpath.data[x+1+(y+1)*m_map_gpath.info.width] = m_map_gpath.data[x+1+(y+1)*m_map_gpath.info.width]<dist/4 ? m_map_gpath.data[x+1+(y+1)*m_map_gpath.info.width]:dist/4;
        m_map_gpath.data[x-1+(y-1)*m_map_gpath.info.width] = m_map_gpath.data[x-1+(y-1)*m_map_gpath.info.width]<dist/4 ? m_map_gpath.data[x-1+(y-1)*m_map_gpath.info.width]:dist/4;
        m_map_gpath.data[x-1+(y+1)*m_map_gpath.info.width] = m_map_gpath.data[x-1+(y+1)*m_map_gpath.info.width]<dist/4 ? m_map_gpath.data[x-1+(y+1)*m_map_gpath.info.width]:dist/4;
        m_map_gpath.data[x+1+(y-1)*m_map_gpath.info.width] = m_map_gpath.data[x+1+(y-1)*m_map_gpath.info.width]<dist/4 ? m_map_gpath.data[x+1+(y-1)*m_map_gpath.info.width]:dist/4;
        dist = std::min(dist+(300/(float)plan.size()), 0.f);
    }
//    m_map_gpath.header.stamp = ros::Time::now();
//    asd.publish(m_map_gpath);
    return true;
}

PLUGINLIB_EXPORT_CLASS(LocalPlanner, nav_core::BaseLocalPlanner)
