#ifndef ROBOTTRAINER_H
#define ROBOTTRAINER_H

#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <tf/transform_datatypes.h>
#include <nav_msgs/OccupancyGrid.h>


class RobotTrainer
{
    typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> movebase_act_client;
public:
    RobotTrainer();
    ~RobotTrainer();

private:
    void new_goal();
    void handle_move_done(const actionlib::SimpleClientGoalState& state,
                          const move_base_msgs::MoveBaseResultConstPtr& result);
    void handle_move_active();
    void handle_move_feedback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback);
    void handle_costmap(const nav_msgs::OccupancyGridConstPtr& msg);
    void handle_timer_traj(const ros::TimerEvent& event);

    ros::NodeHandle m_nh;
    ros::Subscriber m_sub_costmap;
    movebase_act_client m_client_move;
    move_base_msgs::MoveBaseGoal m_goal_move;
    ros::Timer m_timer_traj;

    nav_msgs::OccupancyGrid m_costmap;
};

#endif // ROBOTTRAINER_H
