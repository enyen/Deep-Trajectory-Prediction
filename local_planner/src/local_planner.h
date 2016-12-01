#ifndef LOCALPLANNER_H
#define LOCALPLANNER_H

#include <ros/ros.h>
#include <nav_core/base_local_planner.h>

class LocalPlanner : public nav_core::BaseLocalPlanner
{
public:
    LocalPlanner();
    ~LocalPlanner();

    void initialize(std::string name, tf::TransformListener *tf, costmap_2d::Costmap2DROS *costmap_ros);
    bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);
    bool isGoalReached();
    bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan);

private:

};

#endif // LOCALPLANNER_H
