#ifndef LOCALPLANNER_H
#define LOCALPLANNER_H

#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <nav_core/base_local_planner.h>
#include <costmap_2d/costmap_2d.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>


class LocalPlanner : public nav_core::BaseLocalPlanner
{
//    typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXi8;
    typedef Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic> MatrixXi8;
public:
    LocalPlanner();
    ~LocalPlanner();

    void initialize(std::string name, tf::TransformListener *tf, costmap_2d::Costmap2DROS *costmap_ros);
    bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);
    bool isGoalReached();
    bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan);

private:
    void handle_map_obstacle(const nav_msgs::OccupancyGridConstPtr& msg);
    void handle_map_global(const nav_msgs::OccupancyGridConstPtr& msg);

    ros::NodeHandle m_nh;
    ros::Subscriber m_sub_map_obstacle;
    ros::Subscriber m_sub_map_global;
    ros::Publisher asd;

    tf::TransformListener* m_tf;
    costmap_2d::Costmap2DROS* m_costmap_ros;
    nav_msgs::OccupancyGrid m_map_obstacle;
    MatrixXi8 m_mat_obstacle;
    nav_msgs::OccupancyGrid m_map_global;
    MatrixXi8 m_mat_global;
    nav_msgs::OccupancyGrid m_map_gpath;
};

#endif // LOCALPLANNER_H
