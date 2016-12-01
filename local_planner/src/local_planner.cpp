#include "local_planner.h"
#include <pluginlib/class_list_macros.h>

LocalPlanner::LocalPlanner()
{

}

LocalPlanner::~LocalPlanner()
{

}

void LocalPlanner::initialize(std::string name,
                              tf::TransformListener *tf,
                              costmap_2d::Costmap2DROS *costmap_ros)
{

}

bool LocalPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
{
    return false;
}

bool LocalPlanner::isGoalReached()
{
    return false;
}

bool LocalPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan)
{
    return false;
}

PLUGINLIB_EXPORT_CLASS(LocalPlanner, nav_core::BaseLocalPlanner)
